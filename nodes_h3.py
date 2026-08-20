from .lm_client import (
    run_llm,
    load_guide,
    build_system_message,
    image_tensor_to_base64_url,
    make_contact_sheet,
    contact_sheet_to_base64_url,
    build_settings_brief,
    estimate_tokens
)
from .prompt_loader import SYSTEM_PROMPTS

COMMON_CREATIVE_INPUTS = {
    "has_dialogue": ("BOOLEAN", {"default": False}),
    "dialogue_language": ("STRING", {"default": "English"}),
    "shot_structure": (["auto", "single continuous shot", "multi-shot"], {"default": "auto"}),
    "num_shots": ("INT", {"default": 0, "min": 0, "max": 20}),
    "duration_seconds": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 15.0, "step": 0.5}),
    "visual_style": (["Auto (from brief)", "Cinematic", "Live-action", "2D-animated", "3D CG", "Claymation", "Watercolor", "Vintage film"], {"default": "Auto (from brief)"}),
    "has_music": ("BOOLEAN", {"default": False}),
    "music_description": ("STRING", {"multiline": True, "default": ""}),
    "has_ambience": ("BOOLEAN", {"default": True}),
    "ambience_description": ("STRING", {"multiline": True, "default": ""}),
    "camera_motion_hint": ("STRING", {"multiline": True, "default": ""}),
    "on_screen_text": ("STRING", {"multiline": True, "default": ""}),
    "negative_instructions": ("STRING", {"multiline": True, "default": ""}),
    "extra_instructions": ("STRING", {"multiline": True, "default": ""}),
    "guide_mode": (["compact", "full"], {"default": "compact"}),
    "max_image_dimension": ("INT", {"default": 768, "min": 256, "max": 2048}),
    "assume_context_size": ("INT", {"default": 8192, "min": 1024, "max": 131072}),
    "backend": ("LMC_BACKEND",),
    "base_url": ("STRING", {"default": "http://localhost:1234/v1"}),
    "model": ("STRING", {"default": ""}),
    "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
    "max_tokens": ("INT", {"default": 2500, "min": 1, "max": 16384}),
}

DEFAULT_BASE_BEHAVIOR = SYSTEM_PROMPTS.get("h3_base_behavior", "")

def _context_warning(estimated, max_tokens, assume_context_size):
    if estimated + max_tokens > assume_context_size:
        print(
            f"\n[LM Connect Warning] Tahmini prompt boyutu (~{estimated} token) + max_tokens ({max_tokens}), "
            f"varsayılan context sınırını ({assume_context_size}) aşabilir. "
            "guide_mode='compact' yap, max_image_dimension'ı düşür, veya LM Studio'da modelin "
            "context length ayarını artır.\n"
        )
    return ""

class LMConnectH3Prompt:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "user_brief": ("STRING", {"multiline": True}),
                "mode": (["T2VA - text only", "I2VA - first frame", "FL2VA - first and last frame", "L2VA - last frame only"],),
            },
            "optional": {
                "first_frame_image": ("IMAGE",),
                "last_frame_image": ("IMAGE",),
                "system_prompt_override": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt_addition": ("STRING", {"default": ""}),
            }
        }
        inputs["optional"].update(COMMON_CREATIVE_INPUTS)
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("h3_prompt",)
    FUNCTION = "generate"
    CATEGORY = "LM Connect/MiniMax H3"

    def generate(self, user_brief: str, mode: str,
                 first_frame_image=None, last_frame_image=None,
                 system_prompt_override="", system_prompt_addition="",
                 has_dialogue=False, dialogue_language="English", shot_structure="auto",
                 num_shots=0, duration_seconds=6.0, visual_style="Auto (from brief)",
                 has_music=False, music_description="", has_ambience=True,
                 ambience_description="", camera_motion_hint="", on_screen_text="",
                 negative_instructions="", extra_instructions="",
                 guide_mode="compact", max_image_dimension=768, assume_context_size=8192,
                 backend=None, base_url="http://localhost:1234/v1", model="", temperature=0.4, max_tokens=2500):

        if mode.startswith("I2VA") and first_frame_image is None:
            return ("[LM Connect Error] Mode I2VA requires first_frame_image to be connected.",)
        if mode.startswith("FL2VA") and (first_frame_image is None or last_frame_image is None):
            return ("[LM Connect Error] Mode FL2VA requires both first_frame_image and last_frame_image.",)
        if mode.startswith("L2VA") and last_frame_image is None:
            return ("[LM Connect Error] Mode L2VA requires last_frame_image to be connected.",)

        guide_text = load_guide("base_guide", mode=guide_mode)
        if not guide_text:
            return ("[LM Connect Error] base_guide.md not found in guides folder.",)

        system_message = build_system_message(
            guide_text,
            system_prompt_override if system_prompt_override.strip() else DEFAULT_BASE_BEHAVIOR,
            system_prompt_addition
        )

        settings_brief = build_settings_brief(
            has_dialogue, dialogue_language, shot_structure, num_shots, duration_seconds,
            visual_style, has_music, music_description, has_ambience, ambience_description,
            camera_motion_hint, on_screen_text, negative_instructions, extra_instructions
        )

        user_content = [
            {"type": "text", "text": f"Mode: {mode}\n\nUser brief:\n{user_brief}\n\nCreative settings:\n{settings_brief}"}
        ]

        if first_frame_image is not None:
            user_content.append({"type": "text", "text": "The following image is Picture 1, the reference first frame."})
            user_content.append({"type": "image_url", "image_url": {"url": image_tensor_to_base64_url(first_frame_image, max_dimension=max_image_dimension)}})

        if last_frame_image is not None:
            if mode.startswith("FL2VA"):
                user_content.append({"type": "text", "text": "The following image is Picture 2, the reference last frame."})
            else:
                user_content.append({"type": "text", "text": "The following image is Picture 1, the reference last frame."})
            user_content.append({"type": "image_url", "image_url": {"url": image_tensor_to_base64_url(last_frame_image, max_dimension=max_image_dimension)}})

        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_content}]

        warning = _context_warning(estimate_tokens(messages), max_tokens, assume_context_size)

        response = run_llm(messages, backend=backend, legacy_base_url=base_url, legacy_model=model, legacy_temperature=temperature, legacy_max_tokens=max_tokens)

        return (warning + response,)


DEFAULT_REF_BEHAVIOR = SYSTEM_PROMPTS.get("h3_ref_behavior", "")

class LMConnectH3PromptFullReference:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "user_brief": ("STRING", {"multiline": True}),
                "reference_descriptions": ("STRING", {"multiline": True}),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
                "reference_image_5": ("IMAGE",),
                "reference_video_1": ("IMAGE",),
                "reference_video_2": ("IMAGE",),
                "video_contact_sheet_frames": ("INT", {"default": 9, "min": 4, "max": 16}),
                "audio_references_description": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt_override": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt_addition": ("STRING", {"default": ""}),
            }
        }
        inputs["optional"].update(COMMON_CREATIVE_INPUTS)
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("h3_prompt",)
    FUNCTION = "generate"
    CATEGORY = "LM Connect/MiniMax H3"

    def generate(self, user_brief: str, reference_descriptions: str,
                 reference_image_1=None, reference_image_2=None, reference_image_3=None,
                 reference_image_4=None, reference_image_5=None,
                 reference_video_1=None, reference_video_2=None,
                 video_contact_sheet_frames=9, audio_references_description="",
                 system_prompt_override="", system_prompt_addition="",
                 has_dialogue=False, dialogue_language="English", shot_structure="auto",
                 num_shots=0, duration_seconds=6.0, visual_style="Auto (from brief)",
                 has_music=False, music_description="", has_ambience=True,
                 ambience_description="", camera_motion_hint="", on_screen_text="",
                 negative_instructions="", extra_instructions="",
                 guide_mode="compact", max_image_dimension=768, assume_context_size=8192,
                 backend=None, base_url="http://localhost:1234/v1", model="", temperature=0.4, max_tokens=2500):

        guide_text = load_guide("ref_guide", mode=guide_mode)
        if not guide_text:
            return ("[LM Connect Error] ref_guide.md not found in guides folder.",)

        system_message = build_system_message(
            guide_text,
            system_prompt_override if system_prompt_override.strip() else DEFAULT_REF_BEHAVIOR,
            system_prompt_addition
        )

        settings_brief = build_settings_brief(
            has_dialogue, dialogue_language, shot_structure, num_shots, duration_seconds,
            visual_style, has_music, music_description, has_ambience, ambience_description,
            camera_motion_hint, on_screen_text, negative_instructions, extra_instructions
        )

        user_content = [
            {"type": "text", "text": f"User brief:\n{user_brief}\n\nReference role descriptions (assigned by the user):\n{reference_descriptions}\n\nAudio reference descriptions (text-only, not heard by you):\n{audio_references_description or 'None'}\n\nCreative settings:\n{settings_brief}"}
        ]

        images = [reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5]
        for i, img in enumerate(images):
            if img is not None:
                user_content.append({"type": "text", "text": f"The following image is Picture {i+1}."})
                user_content.append({"type": "image_url", "image_url": {"url": image_tensor_to_base64_url(img, max_dimension=max_image_dimension)}})

        videos = [reference_video_1, reference_video_2]
        for i, vid in enumerate(videos):
            if vid is not None:
                sheet = make_contact_sheet(vid, max_frames=video_contact_sheet_frames)
                if sheet:
                    url = contact_sheet_to_base64_url(sheet, max_dimension=max_image_dimension)
                    user_content.append({"type": "text", "text": f"The following is a contact sheet of sampled frames from Video {i+1}, showing it over time. This is not a set of target shots."})
                    user_content.append({"type": "image_url", "image_url": {"url": url}})

        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_content}]

        warning = _context_warning(estimate_tokens(messages), max_tokens, assume_context_size)

        response = run_llm(messages, backend=backend, legacy_base_url=base_url, legacy_model=model, legacy_temperature=temperature, legacy_max_tokens=max_tokens)

        return (warning + response,)

class LMConnectExtraSystemPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "extra_system_prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt_addition",)
    FUNCTION = "passthrough"
    CATEGORY = "LM Connect/MiniMax H3"

    def passthrough(self, extra_system_prompt: str):
        return (extra_system_prompt,)
