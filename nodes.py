import base64
import io
import numpy as np
from PIL import Image
import torch
from .lm_client import run_llm, image_tensor_to_base64_url

class LMConnectSimplePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "backend": ("LMC_BACKEND",),
                "base_url": ("STRING", {"default": "http://localhost:1234/v1"}),
                "model": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1200, "min": 1, "max": 16384}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "LM Connect"

    def generate(self, prompt: str, backend=None, base_url: str = "http://localhost:1234/v1", model: str = "", temperature: float = 0.7, max_tokens: int = 1200):
        messages = [{"role": "user", "content": prompt}]
        response = run_llm(messages, backend=backend, legacy_base_url=base_url, legacy_model=model, legacy_temperature=temperature, legacy_max_tokens=max_tokens)
        return (response,)

class LMConnectPromptWithSystem:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "backend": ("LMC_BACKEND",),
                "base_url": ("STRING", {"default": "http://localhost:1234/v1"}),
                "model": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1200, "min": 1, "max": 16384}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "LM Connect"

    def generate(self, system_prompt: str, prompt: str, backend=None, base_url: str = "http://localhost:1234/v1", model: str = "", temperature: float = 0.7, max_tokens: int = 1200):
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = run_llm(messages, backend=backend, legacy_base_url=base_url, legacy_model=model, legacy_temperature=temperature, legacy_max_tokens=max_tokens)
        return (response,)

class LMConnectVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "max_image_dimension": ("INT", {"default": 768, "min": 256, "max": 2048}),
                "backend": ("LMC_BACKEND",),
                "base_url": ("STRING", {"default": "http://localhost:1234/v1"}),
                "model": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1200, "min": 1, "max": 16384}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "LM Connect"

    def generate(self, system_prompt: str, prompt: str,
                 image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
                 max_image_dimension=768,
                 backend=None, base_url: str = "http://localhost:1234/v1", model: str = "", temperature: float = 0.7, max_tokens: int = 1200):

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        if prompt and prompt.strip():
            user_content.append({"type": "text", "text": prompt})

        images = [image_1, image_2, image_3, image_4, image_5]
        for img in images:
            if img is not None:
                data_url = image_tensor_to_base64_url(img, max_dimension=max_image_dimension)
                user_content.append({"type": "image_url", "image_url": {"url": data_url}})

        messages.append({"role": "user", "content": user_content})

        response = run_llm(messages, backend=backend, legacy_base_url=base_url, legacy_model=model, legacy_temperature=temperature, legacy_max_tokens=max_tokens)
        return (response,)
