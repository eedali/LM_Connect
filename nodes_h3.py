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

DEFAULT_SWAP_BEHAVIOR = SYSTEM_PROMPTS.get("h3_swap_behavior", "")

class LMConnectH3PersonSwap:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "composition_image": ("IMAGE",),
                "user_brief": ("STRING", {"multiline": True}),
                "reference_descriptions": ("STRING", {"multiline": True}),
            },
            "optional": {
                # --- Reference Person Slot 1 ---
                "ref_image_1": ("IMAGE",),
                "replaces_who_1": ("STRING", {"default": ""}),
                # --- Reference Person Slot 2 ---
                "ref_image_2": ("IMAGE",),
                "replaces_who_2": ("STRING", {"default": ""}),
                # --- Reference Person Slot 3 ---
                "ref_image_3": ("IMAGE",),
                "replaces_who_3": ("STRING", {"default": ""}),
                # --- Reference Person Slot 4 ---
                "ref_image_4": ("IMAGE",),
                "replaces_who_4": ("STRING", {"default": ""}),
                # --- Reference Person Slot 5 ---
                "ref_image_5": ("IMAGE",),
                "replaces_who_5": ("STRING", {"default": ""}),
                # --- Swap Fine-Tuning Controls ---
                "use_composition_clothing": ("BOOLEAN", {"default": True}),
                "use_composition_body_type": ("BOOLEAN", {"default": False}),
                "use_composition_pose": ("BOOLEAN", {"default": True}),
                "use_composition_hairstyle": ("BOOLEAN", {"default": False}),
                "transfer_facial_expression": ("BOOLEAN", {"default": True}),
                "preserve_skin_tone": ("BOOLEAN", {"default": True}),
                "preserve_age_appearance": ("BOOLEAN", {"default": True}),
                "preserve_accessories": ("BOOLEAN", {"default": False}),
                "preserve_scene_lighting": ("BOOLEAN", {"default": True}),
                "preserve_background": ("BOOLEAN", {"default": True}),
                "gender_match_mode": (["auto", "force_ref", "force_comp"], {"default": "auto"}),
                "swap_detail_level": (["minimal", "balanced", "detailed"], {"default": "balanced"}),
                "nsfw": ("BOOLEAN", {"default": False}),
                "analyze_each_image_first": ("BOOLEAN", {"default": False}),
                # --- System Prompt Override ---
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

    def _build_swap_config_brief(self, reference_descriptions, replaces_who_list, use_composition_clothing, use_composition_body_type,
                                  use_composition_pose, use_composition_hairstyle, transfer_facial_expression,
                                  preserve_skin_tone, preserve_age_appearance, preserve_accessories,
                                  preserve_scene_lighting, preserve_background, gender_match_mode,
                                  swap_detail_level):
        lines = []
        lines.append("=== SWAP SLOT ASSIGNMENTS ===")
        lines.append(reference_descriptions)

        has_replaces = any(r.strip() for r in replaces_who_list)
        if has_replaces:
            lines.append("\n=== EXPLICIT REPLACEMENT MAPPINGS ===")
            for i, replaces in enumerate(replaces_who_list, 1):
                if replaces.strip():
                    lines.append(f"- Picture {i} replaces: {replaces.strip()}")

        lines.append("\n=== SWAP CONFIGURATION RULES ===")
        lines.append(f"- clothing_source: {'composition' if use_composition_clothing else 'reference'}")
        lines.append(f"- body_type_source: {'composition' if use_composition_body_type else 'reference'}")
        lines.append(f"- pose_source: {'composition' if use_composition_pose else 'natural'}")
        lines.append(f"- hairstyle_source: {'composition' if use_composition_hairstyle else 'reference'}")
        lines.append(f"- expression_source: {'composition' if transfer_facial_expression else 'natural'}")
        lines.append(f"- preserve_ref_skin_tone: {str(preserve_skin_tone).lower()}")
        lines.append(f"- preserve_ref_age: {str(preserve_age_appearance).lower()}")
        lines.append(f"- preserve_ref_accessories: {str(preserve_accessories).lower()}")
        lines.append(f"- preserve_scene_lighting: {str(preserve_scene_lighting).lower()}")
        lines.append(f"- preserve_background: {str(preserve_background).lower()}")
        lines.append(f"- gender_match_mode: {gender_match_mode}")
        lines.append(f"- swap_detail_level: {swap_detail_level}")

        return "\n".join(lines)

    def generate(self, composition_image, user_brief: str, reference_descriptions: str,
                 ref_image_1=None, replaces_who_1="",
                 ref_image_2=None, replaces_who_2="",
                 ref_image_3=None, replaces_who_3="",
                 ref_image_4=None, replaces_who_4="",
                 ref_image_5=None, replaces_who_5="",
                 use_composition_clothing=True, use_composition_body_type=False,
                 use_composition_pose=True, use_composition_hairstyle=False,
                 transfer_facial_expression=True, preserve_skin_tone=True,
                 preserve_age_appearance=True, preserve_accessories=False,
                 preserve_scene_lighting=True, preserve_background=True,
                 gender_match_mode="auto", swap_detail_level="balanced",
                 nsfw=False, analyze_each_image_first=False,
                 system_prompt_override="", system_prompt_addition="",
                 has_dialogue=False, dialogue_language="English", shot_structure="auto",
                 num_shots=0, duration_seconds=6.0, visual_style="Auto (from brief)",
                 has_music=False, music_description="", has_ambience=True,
                 ambience_description="", camera_motion_hint="", on_screen_text="",
                 negative_instructions="", extra_instructions="",
                 guide_mode="compact", max_image_dimension=768, assume_context_size=8192,
                 backend=None, base_url="http://localhost:1234/v1", model="", temperature=0.4, max_tokens=2500):

        # Collect active swap slots
        active_images = []
        replaces_who_list = []
        
        ref_data = [
            (ref_image_1, replaces_who_1),
            (ref_image_2, replaces_who_2),
            (ref_image_3, replaces_who_3),
            (ref_image_4, replaces_who_4),
            (ref_image_5, replaces_who_5),
        ]
        
        for img, replaces in ref_data:
            if img is not None:
                active_images.append(img)
                replaces_who_list.append(replaces)

        if not active_images:
            return ("[LM Connect Error] At least one reference person image must be connected.",)

        # Load guide and system prompt
        guide_text = load_guide("ref_guide", mode=guide_mode)
        if not guide_text:
            return ("[LM Connect Error] ref_guide.md not found in guides folder.",)

        system_message = build_system_message(
            guide_text,
            system_prompt_override if system_prompt_override.strip() else DEFAULT_SWAP_BEHAVIOR,
            system_prompt_addition
        )

        # Build settings brief
        settings_brief = build_settings_brief(
            has_dialogue, dialogue_language, shot_structure, num_shots, duration_seconds,
            visual_style, has_music, music_description, has_ambience, ambience_description,
            camera_motion_hint, on_screen_text, negative_instructions, extra_instructions
        )

        # Build swap configuration brief
        swap_config = self._build_swap_config_brief(
            reference_descriptions, replaces_who_list, use_composition_clothing, use_composition_body_type,
            use_composition_pose, use_composition_hairstyle, transfer_facial_expression,
            preserve_skin_tone, preserve_age_appearance, preserve_accessories,
            preserve_scene_lighting, preserve_background, gender_match_mode, swap_detail_level
        )

        # Pre-analyze images if requested
        pre_analyzed_texts = []
        if analyze_each_image_first:
            comp_msg = [
                {"role": "system", "content": "You are a helpful assistant. Describe the people and the environment in this image in high detail."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Please describe the characters and the scene in this composition image (Picture COMP)."},
                    {"type": "image_url", "image_url": {"url": image_tensor_to_base64_url(composition_image, max_dimension=max_image_dimension)}}
                ]}
            ]
            comp_desc = run_llm(comp_msg, backend=backend, legacy_base_url=base_url, legacy_model=model, legacy_temperature=temperature, legacy_max_tokens=500)
            pre_analyzed_texts.append(f"--- Picture COMP (Composition Image) Pre-Analyzed Description ---\n{comp_desc}")

            for idx, img in enumerate(active_images, 1):
                ref_msg = [
                    {"role": "system", "content": "You are a helpful assistant. Describe the person/character in the image in high detail (clothing, facial features, hair, accessories). Do not describe the background, only the character."},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Please describe the character in this reference image (Picture {idx})."},
                        {"type": "image_url", "image_url": {"url": image_tensor_to_base64_url(img, max_dimension=max_image_dimension)}}
                    ]}
                ]
                ref_desc = run_llm(ref_msg, backend=backend, legacy_base_url=base_url, legacy_model=model, legacy_temperature=temperature, legacy_max_tokens=500)
                pre_analyzed_texts.append(f"--- Picture {idx} (Reference {idx}) Pre-Analyzed Character Description ---\n{ref_desc}")

        extra_rules = [f"NSFW Allowed: {'YES' if nsfw else 'NO'}"]
        if pre_analyzed_texts:
            extra_rules.append("\n".join(pre_analyzed_texts))

        # Build user content
        user_content = [
            {"type": "text", "text": (
                f"User brief:\n{user_brief}\n\n"
                f"{swap_config}\n\n"
                f"{chr(10).join(extra_rules)}\n\n"
                f"Creative settings:\n{settings_brief}"
            )}
        ]

        # Add composition image
        user_content.append({"type": "text", "text": "The following image is the COMPOSITION IMAGE (Picture COMP) — the scene layout, pose, and environment reference."})
        user_content.append({"type": "image_url", "image_url": {"url": image_tensor_to_base64_url(composition_image, max_dimension=max_image_dimension)}})

        # Add reference person images
        for i, img in enumerate(active_images):
            user_content.append({"type": "text", "text": f"The following image is REFERENCE PERSON {i+1} (Picture {i+1}) — the identity source for swap slot {i+1}."})
            user_content.append({"type": "image_url", "image_url": {"url": image_tensor_to_base64_url(img, max_dimension=max_image_dimension)}})

        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_content}]

        warning = _context_warning(estimate_tokens(messages), max_tokens, assume_context_size)

        response = run_llm(messages, backend=backend, legacy_base_url=base_url, legacy_model=model, legacy_temperature=temperature, legacy_max_tokens=max_tokens)

        return (warning + response,)


DEFAULT_DIRECTOR_BEHAVIOR = SYSTEM_PROMPTS.get("h3_director_behavior", "")

class LMConnectH3ImageToVideoPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "general_prompt": ("STRING", {"multiline": True}),
                "location": (["New Location", "Use Image 1 Location", "Use Image 2 Location", "Use Image 3 Location", "Use Image 4 Location", "Use Image 5 Location"], {"default": "New Location"}),
                "new_location_prompt": ("STRING", {"multiline": True, "default": ""}),
                "nsfw": ("BOOLEAN", {"default": False}),
                "analyze_each_image_first": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
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

    def generate(self, general_prompt: str, location: str, new_location_prompt: str,
                 nsfw: bool, analyze_each_image_first: bool,
                 image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
                 system_prompt_override="", system_prompt_addition="",
                 has_dialogue=False, dialogue_language="English", shot_structure="auto",
                 num_shots=0, duration_seconds=6.0, visual_style="Auto (from brief)",
                 has_music=False, music_description="", has_ambience=True,
                 ambience_description="", camera_motion_hint="", on_screen_text="",
                 negative_instructions="", extra_instructions="",
                 guide_mode="compact", max_image_dimension=768, assume_context_size=8192,
                 backend=None, base_url="http://localhost:1234/v1", model="", temperature=0.4, max_tokens=2500):

        # Collect active images
        active_images = []
        for i, img in enumerate([image_1, image_2, image_3, image_4, image_5]):
            if img is not None:
                active_images.append((i+1, img))

        if not active_images:
            return ("[LM Connect Error] At least one image must be connected.",)

        # Load guide
        guide_text = load_guide("ref_guide", mode=guide_mode)
        if not guide_text:
            return ("[LM Connect Error] ref_guide.md not found in guides folder.",)

        # Pre-analyze images if requested
        pre_analyzed_texts = []
        if analyze_each_image_first:
            for idx, img in active_images:
                analysis_messages = [
                    {"role": "system", "content": "You are a helpful assistant. Describe the person/character in the image in high detail (clothing, facial features, hair, accessories). Do not describe the background, only the character."},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Please describe the character in this image (Picture {idx})."},
                        {"type": "image_url", "image_url": {"url": image_tensor_to_base64_url(img, max_dimension=max_image_dimension)}}
                    ]}
                ]
                desc = run_llm(analysis_messages, backend=backend, legacy_base_url=base_url, legacy_model=model, legacy_temperature=temperature, legacy_max_tokens=500)
                pre_analyzed_texts.append(f"--- Picture {idx} Pre-Analyzed Character Description ---\n{desc}")

        # Build final LLM messages
        system_message = build_system_message(
            guide_text,
            system_prompt_override if system_prompt_override.strip() else DEFAULT_DIRECTOR_BEHAVIOR,
            system_prompt_addition
        )

        settings_brief = build_settings_brief(
            has_dialogue, dialogue_language, shot_structure, num_shots, duration_seconds,
            visual_style, has_music, music_description, has_ambience, ambience_description,
            camera_motion_hint, on_screen_text, negative_instructions, extra_instructions
        )
        
        brief_parts = [f"General Prompt:\n{general_prompt}"]
        if location == "New Location":
            brief_parts.append(f"Location Rule: Use the following new location description:\n{new_location_prompt}")
        else:
            brief_parts.append(f"Location Rule: Match the environment and background from {location.replace('Use ', '')}.")
            
        brief_parts.append(f"NSFW Allowed: {'YES' if nsfw else 'NO'}")
        
        if pre_analyzed_texts:
            brief_parts.append("\n".join(pre_analyzed_texts))

        user_content = [
            {"type": "text", "text": (
                f"User Instructions & Context:\n"
                f"{chr(10).join(brief_parts)}\n\n"
                f"Creative settings:\n{settings_brief}"
            )}
        ]

        # Add images
        for idx, img in active_images:
            user_content.append({"type": "text", "text": f"The following image is Picture {idx}."})
            user_content.append({"type": "image_url", "image_url": {"url": image_tensor_to_base64_url(img, max_dimension=max_image_dimension)}})

        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_content}]
        warning = _context_warning(estimate_tokens(messages), max_tokens, assume_context_size)
        response = run_llm(messages, backend=backend, legacy_base_url=base_url, legacy_model=model, legacy_temperature=temperature, legacy_max_tokens=max_tokens)

        return (warning + response,)


DEFAULT_ORBIT_BEHAVIOR = SYSTEM_PROMPTS.get("h3_orbit_behavior", "")

IMAGE_ROLES = ["Character Reference", "Object Reference", "Background / Location Reference"]

BACKGROUND_PRESETS = [
    "Pure White (Studio)",
    "Studio Gray (Neutral)",
    "Black Void",
    "Outdoor / Natural Light",
    "Custom",
]

FRAMING_OPTIONS = [
    "Full Body (head to toe)",
    "3/4 Body (head to knees)",
    "Waist Up (medium shot)",
    "Chest Up (bust shot)",
    "Head & Shoulders (close-up)",
    "Face Only (extreme close-up)",
]

POSE_PRESETS = [
    "Custom (describe below)",
    # --- Classic 3D / Photogrammetry ---
    "T-Pose (arms straight out horizontally)",
    "A-Pose (arms slightly lowered at 45 degrees)",
    # --- Standing ---
    "Standing straight, arms at sides",
    "Standing with arms crossed",
    "Standing with hands on hips",
    "Standing contrapposto (weight on one leg)",
    "Standing with legs apart, power stance",
    "Standing with hands in pockets",
    "Standing with one hand raised (waving)",
    "Standing back-to-back (multiple characters)",
    # --- Sitting ---
    "Sitting on a chair, hands on knees",
    "Sitting cross-legged on the floor",
    "Sitting on the ground, legs extended",
    "Sitting with one leg crossed over the other",
    "Sitting on the edge of a surface, legs dangling",
    # --- Kneeling / Crouching ---
    "Kneeling on one knee",
    "Kneeling on both knees, upright",
    "Crouching / Squatting",
    # --- Action Freeze ---
    "Walking mid-stride (frozen)",
    "Running mid-stride (frozen)",
    "Jumping (frozen mid-air)",
    "Fighting stance (guard up)",
    "Martial arts kick (frozen mid-kick)",
    "Punching forward (frozen mid-swing)",
    "Dancing (frozen mid-move)",
    "Throwing (frozen mid-throw)",
    # --- Fashion / Modeling ---
    "Fashion pose (hand on hip, slight body turn)",
    "Profile view (body turned 90 degrees to the side)",
    "Back turned (facing away from camera start)",
    "Looking over shoulder",
    "Arms raised above head",
    "Leaning against a wall or surface",
    "Model walk (frozen mid-catwalk stride)",
    # --- Reclining / Lying ---
    "Lying down on back (supine)",
    "Lying on side (recumbent)",
    "Lying face down (prone)",
]

class LMConnectH3OrbitShot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "framing": (FRAMING_OPTIONS, {"default": "Full Body (head to toe)"}),
                "pose_preset": (POSE_PRESETS, {"default": "Standing straight, arms at sides"}),
                "custom_pose_prompt": ("STRING", {"multiline": True, "default": ""}),
                "background_preset": (BACKGROUND_PRESETS, {"default": "Pure White (Studio)"}),
                "custom_background_prompt": ("STRING", {"multiline": True, "default": ""}),
                "orbit_direction": (["clockwise", "counter-clockwise"], {"default": "clockwise"}),
                "camera_height": (["eye-level", "slightly above", "slightly below", "top-down"], {"default": "eye-level"}),
                "duration_seconds": ("FLOAT", {"default": 6.0, "min": 5.0, "max": 15.0, "step": 0.5}),
                "nsfw": ("BOOLEAN", {"default": False}),
                "analyze_each_image_first": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # --- Image Slots with Role Assignment ---
                "image_1": ("IMAGE",),
                "image_role_1": (IMAGE_ROLES, {"default": "Character Reference"}),
                "image_2": ("IMAGE",),
                "image_role_2": (IMAGE_ROLES, {"default": "Character Reference"}),
                "image_3": ("IMAGE",),
                "image_role_3": (IMAGE_ROLES, {"default": "Character Reference"}),
                "image_4": ("IMAGE",),
                "image_role_4": (IMAGE_ROLES, {"default": "Character Reference"}),
                "image_5": ("IMAGE",),
                "image_role_5": (IMAGE_ROLES, {"default": "Character Reference"}),
                # --- Visual & Creative ---
                "visual_style": (["Auto (from brief)", "Cinematic", "Live-action", "2D-animated", "3D CG", "Claymation", "Watercolor", "Vintage film"], {"default": "Live-action"}),
                "negative_instructions": ("STRING", {"multiline": True, "default": ""}),
                "extra_instructions": ("STRING", {"multiline": True, "default": ""}),
                # --- System Prompt Override ---
                "system_prompt_override": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt_addition": ("STRING", {"default": ""}),
                # --- LLM Connection ---
                "guide_mode": (["compact", "full"], {"default": "compact"}),
                "max_image_dimension": ("INT", {"default": 768, "min": 256, "max": 2048}),
                "assume_context_size": ("INT", {"default": 8192, "min": 1024, "max": 131072}),
                "backend": ("LMC_BACKEND",),
                "base_url": ("STRING", {"default": "http://localhost:1234/v1"}),
                "model": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2500, "min": 1, "max": 16384}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("h3_prompt",)
    FUNCTION = "generate"
    CATEGORY = "LM Connect/MiniMax H3"

    def generate(self, framing: str, pose_preset: str, custom_pose_prompt: str,
                 background_preset: str, custom_background_prompt: str,
                 orbit_direction: str, camera_height: str, duration_seconds: float,
                 nsfw: bool, analyze_each_image_first: bool,
                 image_1=None, image_role_1="Character Reference",
                 image_2=None, image_role_2="Character Reference",
                 image_3=None, image_role_3="Character Reference",
                 image_4=None, image_role_4="Character Reference",
                 image_5=None, image_role_5="Character Reference",
                 visual_style="Live-action",
                 negative_instructions="", extra_instructions="",
                 system_prompt_override="", system_prompt_addition="",
                 guide_mode="compact", max_image_dimension=768, assume_context_size=8192,
                 backend=None, base_url="http://localhost:1234/v1", model="", temperature=0.4, max_tokens=2500):

        # Collect active images with their roles
        all_slots = [
            (image_1, image_role_1),
            (image_2, image_role_2),
            (image_3, image_role_3),
            (image_4, image_role_4),
            (image_5, image_role_5),
        ]
        active_images = []  # list of (slot_index, image_tensor, role)
        for i, (img, role) in enumerate(all_slots):
            if img is not None:
                active_images.append((i + 1, img, role))

        if not active_images:
            return ("[LM Connect Error] At least one image must be connected.",)

        # Load guide
        guide_text = load_guide("ref_guide", mode=guide_mode)
        if not guide_text:
            return ("[LM Connect Error] ref_guide.md not found in guides folder.",)

        # Pre-analyze images if requested
        pre_analyzed_texts = []
        if analyze_each_image_first:
            for idx, img, role in active_images:
                if role == "Character Reference":
                    sys_msg = "You are a helpful assistant. Describe the person/character in the image in high detail: face shape, skin tone, hair color/style/length, eye color, build/physique, clothing (every garment, color, texture), accessories (glasses, jewelry, watch, hat), footwear, and any distinguishing features (tattoos, scars, birthmarks). Do NOT describe the background."
                    usr_msg = f"Please describe the character in this image (Picture {idx}) in exhaustive detail."
                elif role == "Object Reference":
                    sys_msg = "You are a helpful assistant. Describe the object in the image in high detail: shape, dimensions, material, color, texture, surface finish, notable features, and any text/labels visible on it. Do NOT describe the background."
                    usr_msg = f"Please describe the object in this image (Picture {idx}) in exhaustive detail."
                else:  # Background / Location Reference
                    sys_msg = "You are a helpful assistant. Describe the environment/location in the image in high detail: setting type, lighting conditions, floor/ground material, walls/sky, colors, atmosphere, and any notable environmental features."
                    usr_msg = f"Please describe the environment/location in this image (Picture {idx}) in detail."

                analysis_messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": [
                        {"type": "text", "text": usr_msg},
                        {"type": "image_url", "image_url": {"url": image_tensor_to_base64_url(img, max_dimension=max_image_dimension)}}
                    ]}
                ]
                desc = run_llm(analysis_messages, backend=backend, legacy_base_url=base_url, legacy_model=model, legacy_temperature=temperature, legacy_max_tokens=500)
                pre_analyzed_texts.append(f"--- Picture {idx} ({role}) Pre-Analyzed Description ---\n{desc}")

        # Build system message
        system_message = build_system_message(
            guide_text,
            system_prompt_override if system_prompt_override.strip() else DEFAULT_ORBIT_BEHAVIOR,
            system_prompt_addition
        )

        # Build hardcoded settings brief (no dialogue, no music, no ambience, single shot)
        settings_lines = [
            f"- Target duration: {duration_seconds:.1f} seconds.",
            "- Shot structure: single continuous shot.",
            "- Dialogue: no.",
            "- Music: no.",
            "- Ambience: no.",
        ]
        if visual_style != "Auto (from brief)":
            settings_lines.append(f"- Visual style: {visual_style}.")
        settings_lines.append(f"- Camera motion hint: Smooth {orbit_direction} 360-degree orbit at {camera_height} height, steady mechanical pace, always facing center of subject(s).")
        if negative_instructions.strip():
            settings_lines.append(f"- Avoid: {negative_instructions.strip()}")
        if extra_instructions.strip():
            settings_lines.append(f"- Extra notes: {extra_instructions.strip()}")
        settings_brief = "\n".join(settings_lines)

        # Build image role summary
        role_lines = ["=== IMAGE ROLE ASSIGNMENTS ==="]
        for idx, img, role in active_images:
            role_lines.append(f"- Picture {idx}: {role}")

        # Build background info
        bg_lines = ["=== BACKGROUND SETTING ==="]
        if background_preset == "Custom":
            bg_lines.append(f"Background: Custom — {custom_background_prompt}")
        else:
            bg_lines.append(f"Background: {background_preset}")

        # Build pose info
        pose_lines = ["=== POSE ==="]
        if pose_preset == "Custom (describe below)":
            pose_lines.append(f"Pose: Custom — {custom_pose_prompt}")
        else:
            pose_lines.append(f"Pose Preset: {pose_preset}")
            if custom_pose_prompt.strip():
                pose_lines.append(f"Additional Pose Notes: {custom_pose_prompt.strip()}")

        # Build full user content text
        brief_parts = [
            f"Framing: {framing}",
            "\n".join(pose_lines),
            "\n".join(role_lines),
            "\n".join(bg_lines),
            f"Orbit Direction: {orbit_direction}",
            f"Camera Height: {camera_height}",
            f"NSFW Allowed: {'YES' if nsfw else 'NO'}",
        ]

        if pre_analyzed_texts:
            brief_parts.append("\n".join(pre_analyzed_texts))

        user_content = [
            {"type": "text", "text": (
                f"User Instructions & Context:\n"
                f"{chr(10).join(brief_parts)}\n\n"
                f"Creative settings:\n{settings_brief}"
            )}
        ]

        # Add images with role labels
        for idx, img, role in active_images:
            user_content.append({"type": "text", "text": f"The following image is Picture {idx} (Role: {role})."})
            user_content.append({"type": "image_url", "image_url": {"url": image_tensor_to_base64_url(img, max_dimension=max_image_dimension)}})

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
