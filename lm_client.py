import requests
import json
import time
from urllib.parse import urlparse
from typing import List, Dict, Any

_LOCAL_MODEL_CACHE = {}  # key -> llama_cpp.Llama instance

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(messages) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += len(part.get("text", "")) // 4
                elif part.get("type") == "image_url":
                    total += 600
    return total

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def image_tensor_to_base64_url(image_tensor, max_dimension: int = 768) -> str:
    import base64
    import io
    import numpy as np
    from PIL import Image as PILImage
    if len(image_tensor.shape) == 4:
        img = image_tensor[0]
    else:
        img = image_tensor
    img = 255. * img.cpu().numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil_image = PILImage.fromarray(img)
    if max_dimension and max_dimension > 0:
        pil_image.thumbnail((max_dimension, max_dimension), PILImage.LANCZOS)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def make_contact_sheet(image_batch_tensor, max_frames: int = 9, grid_cols: int = 3):
    from PIL import Image as PILImage
    import numpy as np

    num_frames = image_batch_tensor.shape[0]
    if num_frames == 0:
        return None

    if num_frames <= max_frames:
        indices = np.arange(num_frames)
    else:
        indices = np.linspace(0, num_frames - 1, max_frames, dtype=int)

    sampled_frames = image_batch_tensor[indices]

    pil_images = []
    for i in range(sampled_frames.shape[0]):
        img = sampled_frames[i]
        img = 255. * img.cpu().numpy()
        img = np.clip(img, 0, 255).astype(np.uint8)
        pil_images.append(PILImage.fromarray(img))

    if not pil_images:
        return None

    width, height = pil_images[0].size
    num_sampled = len(pil_images)

    grid_rows = (num_sampled + grid_cols - 1) // grid_cols
    sep = 2
    grid_width = grid_cols * width + (grid_cols - 1) * sep
    grid_height = grid_rows * height + (grid_rows - 1) * sep

    contact_sheet = PILImage.new('RGB', (grid_width, grid_height), color='white')

    for idx, img in enumerate(pil_images):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * (width + sep)
        y = row * (height + sep)
        contact_sheet.paste(img, (x, y))

    return contact_sheet

def contact_sheet_to_base64_url(contact_sheet, max_dimension: int = 768) -> str:
    import base64
    import io
    from PIL import Image as PILImage
    if contact_sheet is None:
        return ""
    if max_dimension and max_dimension > 0:
        contact_sheet.thumbnail((max_dimension, max_dimension), PILImage.LANCZOS)
    buffered = io.BytesIO()
    contact_sheet.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# ---------------------------------------------------------------------------
# Guide loading
# ---------------------------------------------------------------------------

def load_guide(name: str, mode: str = "full") -> str:
    import os
    guides_dir = os.path.join(os.path.dirname(__file__), "guides")
    if mode == "compact":
        compact_path = os.path.join(guides_dir, f"{name}_compact.md")
        if os.path.exists(compact_path):
            with open(compact_path, "r", encoding="utf-8") as f:
                return f.read()
        print(f"[LM Connect Warning] Compact guide not found ({compact_path}), falling back to full guide.")
    full_path = os.path.join(guides_dir, f"{name}.md")
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    print(f"[LM Connect Warning] Guide file not found: {full_path}")
    return ""

def build_system_message(guide_text: str, behavior_prompt: str, addition: str) -> str:
    msg = f"## OFFICIAL MINIMAX H3 PROMPT WRITING GUIDE (must be followed exactly)\n\n{guide_text}\n\n## BEHAVIOR INSTRUCTIONS\n\n{behavior_prompt}"
    if addition and addition.strip():
        msg += f"\n\n## ADDITIONAL SYSTEM INSTRUCTIONS\n\n{addition}"
    return msg

def build_settings_brief(has_dialogue, dialogue_language, shot_structure, num_shots, duration_seconds, visual_style, has_music, music_description, has_ambience, ambience_description, camera_motion_hint, on_screen_text, negative_instructions, extra_instructions) -> str:
    lines = []
    lines.append(f"- Target duration: {duration_seconds:.1f} seconds.")
    if shot_structure == "multi-shot" and num_shots > 0:
        lines.append(f"- Shot structure: {shot_structure} (target {num_shots} shots).")
    elif shot_structure != "auto":
        lines.append(f"- Shot structure: {shot_structure}.")
    if has_dialogue:
        lines.append(f"- Dialogue: yes, in {dialogue_language}.")
    else:
        lines.append("- Dialogue: no.")
    if visual_style != "Auto (from brief)":
        lines.append(f"- Visual style: {visual_style}.")
    if has_music:
        md = f" - {music_description.strip()}" if music_description.strip() else ""
        lines.append(f"- Music: yes{md}")
    else:
        lines.append("- Music: no.")
    if has_ambience:
        ad = f" - {ambience_description.strip()}" if ambience_description.strip() else ""
        lines.append(f"- Ambience: yes{ad}")
    else:
        lines.append("- Ambience: no.")
    if camera_motion_hint.strip():
        lines.append(f"- Camera motion hint: {camera_motion_hint.strip()}")
    if on_screen_text.strip():
        lines.append(f"- On-screen text: {on_screen_text.strip()}")
    if negative_instructions.strip():
        lines.append(f"- Avoid: {negative_instructions.strip()}")
    if extra_instructions.strip():
        lines.append(f"- Extra notes: {extra_instructions.strip()}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Thinking-mode empty content detection
# ---------------------------------------------------------------------------

def _check_thinking_empty(choice) -> str | None:
    """Return an error string if the model spent all tokens on reasoning and produced no content."""
    message = choice.get("message", {})
    content = (message.get("content") or "").strip()
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    finish_reason = choice.get("finish_reason")

    if not content and reasoning:
        preview = reasoning[-400:]
        return (
            "[LM Connect Error] Model 'thinking/reasoning' modunda tüm token bütçesini "
            f"iç muhakemede tüketti ve nihai cevabı yazamadı (finish_reason={finish_reason}).\n"
            "Şunlardan birini dene: max_tokens'i artır (örn. 2500+), disable_thinking'i açık bırak, "
            "veya 'thinking/reasoning' etiketi olmayan bir model kullan.\n"
            f"--- Modelin yarım kalan düşüncesinin sonu ---\n{preview}"
        )
    return None

# ---------------------------------------------------------------------------
# LM Studio: Eject (Native REST API)
# ---------------------------------------------------------------------------

def eject_lmstudio_model(base_url, model_key=None, api_key="lm-studio", debug=False):
    parsed = urlparse(base_url)
    native_base = f"{parsed.scheme}://{parsed.netloc}/api/v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(f"{native_base}/models", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"[LM Connect Warning] Model listesi alınamadı: {e}"

    if debug:
        print(f"[LM Connect DEBUG] GET {native_base}/models -> {data}")

    entries = data.get("data") or data.get("models") or (data if isinstance(data, list) else [])
    instance_ids = []
    for entry in entries:
        entry_key = entry.get("id") or entry.get("key") or entry.get("modelKey")
        if model_key and entry_key != model_key:
            continue
        for inst in entry.get("loaded_instances", entry.get("instances", [])):
            inst_id = inst.get("id") if isinstance(inst, dict) else inst
            if inst_id:
                instance_ids.append(inst_id)

    if not instance_ids:
        return f"[LM Connect Warning] '{model_key or 'any'}' için yüklü instance bulunamadı, eject atlandı."

    results = []
    for inst_id in instance_ids:
        try:
            r = requests.post(f"{native_base}/models/unload", json={"instance_id": inst_id},
                               headers=headers, timeout=15)
            results.append(f"{inst_id}: {r.status_code}")
            if debug:
                print(f"[LM Connect DEBUG] unload {inst_id} -> {r.status_code} {r.text}")
        except Exception as e:
            results.append(f"{inst_id}: ERROR {e}")
    return "[LM Connect] Eject sonucu: " + ", ".join(results)

def load_lmstudio_model(base_url, model_key, api_key="lm-studio", debug=False):
    if not model_key:
        return "[LM Connect Warning] Yüklenecek model belirtilmedi."
    parsed = urlparse(base_url)
    native_base = f"{parsed.scheme}://{parsed.netloc}/api/v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        if debug:
            print(f"[LM Connect DEBUG] POST {native_base}/models/load for {model_key}")
        r = requests.post(f"{native_base}/models/load", json={"model": model_key},
                           headers=headers, timeout=60)
        r.raise_for_status()
        if debug:
            print(f"[LM Connect DEBUG] load -> {r.status_code} {r.text}")
        return f"[LM Connect] Load başarılı: {model_key}"
    except Exception as e:
        return f"[LM Connect Error] Model yüklenemedi ({model_key}): {e}"

# ---------------------------------------------------------------------------
# LM Studio: Advanced streaming call
# ---------------------------------------------------------------------------

def call_lmstudio_advanced(messages, model, temperature, max_tokens, cfg) -> str:
    base_url = cfg.get("base_url", "http://localhost:1234/v1")
    if not base_url.endswith("/"):
        base_url += "/"

    connect_timeout = cfg.get("connect_timeout", 10)
    read_timeout = cfg.get("read_timeout", 600)
    stream = cfg.get("stream", True)
    max_retries = cfg.get("max_retries", 1)
    health_check = cfg.get("health_check", True)
    debug = cfg.get("debug_logging", False)
    disable_thinking = cfg.get("disable_thinking", True)

    if health_check:
        try:
            requests.get(f"{base_url}models", timeout=5)
        except Exception:
            return f"[LM Connect Error] LM Studio'ya ulaşılamıyor, sunucu açık mı? (base_url: {base_url})"

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        payload["model"] = model
    if stream:
        payload["stream"] = True
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    endpoint = f"{base_url}chat/completions"

    attempt = 0
    while attempt <= max_retries:
        try:
            if stream:
                response = requests.post(endpoint, json=payload, stream=True, timeout=(connect_timeout, read_timeout))
                response.raise_for_status()

                content_pieces = []
                reasoning_pieces = []
                finish_reason = None
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            data_str = decoded[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                choice = chunk.get("choices", [{}])[0]
                                delta = choice.get("delta", {})
                                if "content" in delta and delta["content"]:
                                    content_pieces.append(delta["content"])
                                if "reasoning_content" in delta and delta["reasoning_content"]:
                                    reasoning_pieces.append(delta["reasoning_content"])
                                if choice.get("finish_reason"):
                                    finish_reason = choice["finish_reason"]
                            except:
                                pass

                final_text = "".join(content_pieces)
                reasoning_text = "".join(reasoning_pieces)

                if not final_text.strip() and reasoning_text:
                    preview = reasoning_text[-400:]
                    final_text = (
                        "[LM Connect Error] Model 'thinking/reasoning' modunda tüm token bütçesini "
                        f"iç muhakemede tüketti ve nihai cevabı yazamadı (finish_reason={finish_reason}).\n"
                        "Şunlardan birini dene: max_tokens'i artır (örn. 2500+), disable_thinking'i açık bırak, "
                        "veya 'thinking/reasoning' etiketi olmayan bir model kullan.\n"
                        f"--- Modelin yarım kalan düşüncesinin sonu ---\n{preview}"
                    )

                if cfg.get("auto_eject_after_run", False):
                    eject_result = eject_lmstudio_model(base_url, model, debug=debug)
                    if debug:
                        print(f"[LM Connect DEBUG] Auto-eject result: {eject_result}")

                return final_text
            else:
                response = requests.post(endpoint, json=payload, timeout=(connect_timeout, read_timeout))
                response.raise_for_status()
                data = response.json()

                if cfg.get("auto_eject_after_run", False):
                    eject_result = eject_lmstudio_model(base_url, model, debug=debug)
                    if debug:
                        print(f"[LM Connect DEBUG] Auto-eject result: {eject_result}")

                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    thinking_err = _check_thinking_empty(choice)
                    if thinking_err:
                        return thinking_err
                    return choice["message"]["content"]
                else:
                    return f"[LM Connect Error] Unexpected API response format: {json.dumps(data)}"

        except requests.exceptions.ConnectionError:
            return f"[LM Connect Error] LM Studio'ya bağlanılamadı (sunucu kapalı olabilir): {base_url}"
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                attempt += 1
                time.sleep(2 ** attempt)
                continue
            return f"[LM Connect Error] LM Studio {read_timeout} saniyedir hiç yanıt vermedi (model çok mu büyük / GPU meşgul mü kontrol et)."
        except requests.exceptions.HTTPError as e:
            return f"[LM Connect Error] HTTP Error {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"[LM Connect Error] An unexpected error occurred: {str(e)}"

    return "[LM Connect Error] Max retries exceeded."

# ---------------------------------------------------------------------------
# Legacy LM Studio call (no backend connected)
# ---------------------------------------------------------------------------

def call_lm_studio(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: int = 120
) -> str:
    """Legacy method for LM Studio OpenAI-compatible API."""
    if not base_url.endswith("/"):
        base_url += "/"

    endpoint = f"{base_url}chat/completions"

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if model:
        payload["model"] = model

    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return f"[LM Connect Error] Unexpected API response format: {json.dumps(data)}"
    except requests.exceptions.Timeout:
        return "[LM Connect Error] Request to LM Studio timed out."
    except requests.exceptions.ConnectionError:
        return f"[LM Connect Error] Failed to connect to LM Studio at {base_url}. Is the server running?"
    except requests.exceptions.HTTPError as e:
        return f"[LM Connect Error] HTTP Error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"[LM Connect Error] An unexpected error occurred: {str(e)}"

# ---------------------------------------------------------------------------
# Local GGUF (llama-cpp-python)
# ---------------------------------------------------------------------------

def _cache_key(cfg):
    return (
        cfg.get("model_path"),
        cfg.get("mmproj_path") or "",
        cfg.get("n_gpu_layers", -1),
        cfg.get("n_ctx", 8192),
        cfg.get("n_threads", 0),
        cfg.get("chat_format") or "auto"
    )

def get_or_load_local_model(cfg):
    key = _cache_key(cfg)
    if key in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[key]

    from llama_cpp import Llama
    chat_handler = None
    verbose = cfg.get("verbose", False)
    mmproj_path = cfg.get("mmproj_path", "")

    if mmproj_path:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=verbose)

    chat_format = cfg.get("chat_format", "auto")
    if chat_handler:
        final_chat_format = None
    elif chat_format == "auto":
        final_chat_format = None
    else:
        final_chat_format = chat_format

    llm = Llama(
        model_path=cfg["model_path"],
        chat_handler=chat_handler,
        n_gpu_layers=cfg.get("n_gpu_layers", -1),
        n_ctx=cfg.get("n_ctx", 8192),
        n_threads=cfg.get("n_threads", 0) or None,
        chat_format=final_chat_format,
        verbose=verbose,
    )
    _LOCAL_MODEL_CACHE[key] = llm
    return llm

def eject_local_model(cfg=None):
    """cfg verilirse sadece o modeli, verilmezse cache'teki TÜM yerel modelleri VRAM'den atar."""
    import gc
    keys = [_cache_key(cfg)] if cfg else list(_LOCAL_MODEL_CACHE.keys())
    for k in keys:
        llm = _LOCAL_MODEL_CACHE.pop(k, None)
        if llm is not None:
            if hasattr(llm, "close"):
                try:
                    llm.close()
                except:
                    pass
            del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

def call_local_gguf(messages, temperature, max_tokens, cfg) -> str:
    disable_thinking = cfg.get("disable_thinking", True)

    try:
        llm = get_or_load_local_model(cfg)
    except Exception as e:
        return f"[LM Connect Error] Failed to load local GGUF model: {e}"

    extra_kwargs = {}
    if disable_thinking:
        extra_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        result = llm.create_chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens, **extra_kwargs)
        choice = result["choices"][0]
        thinking_err = _check_thinking_empty(choice)
        if thinking_err:
            return thinking_err
        text = choice["message"]["content"]
    except Exception as e:
        return f"[LM Connect Error] Local GGUF inference failed: {e}"
    finally:
        if not cfg.get("keep_loaded", False):
            eject_local_model(cfg)
    return text

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_llm(messages, backend=None, legacy_base_url=None, legacy_model=None,
            legacy_temperature=0.7, legacy_max_tokens=512) -> str:
    if backend is None:
        return call_lm_studio(legacy_base_url or "http://localhost:1234/v1", legacy_model or "", messages, legacy_temperature, legacy_max_tokens)

    backend_type = backend.get("type")

    if backend_type == "lmstudio":
        return call_lmstudio_advanced(messages, legacy_model or backend.get("model", ""), legacy_temperature, legacy_max_tokens, backend)
    elif backend_type == "local_gguf":
        return call_local_gguf(messages, legacy_temperature, legacy_max_tokens, backend)
    else:
        return f"[LM Connect Error] Unknown backend type: {backend_type}"
