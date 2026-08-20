from .lm_client import eject_local_model, eject_lmstudio_model, load_lmstudio_model

class LMConnectLMStudioBackend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": "http://localhost:1234/v1"}),
                "model": ("STRING", {"default": ""}),
            },
            "optional": {
                "connect_timeout_seconds": ("INT", {"default": 10, "min": 1, "max": 120}),
                "read_timeout_seconds": ("INT", {"default": 600, "min": 10, "max": 3600}),
                "stream": ("BOOLEAN", {"default": True}),
                "max_retries": ("INT", {"default": 1, "min": 0, "max": 5}),
                "health_check": ("BOOLEAN", {"default": True}),
                "auto_eject_after_run": ("BOOLEAN", {"default": False}),
                "disable_thinking": ("BOOLEAN", {"default": True}),
                "debug_logging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LMC_BACKEND",)
    RETURN_NAMES = ("backend",)
    FUNCTION = "build_backend"
    CATEGORY = "LM Connect/Backends"

    def build_backend(self, base_url="http://localhost:1234/v1", model="", connect_timeout_seconds=10,
                      read_timeout_seconds=600, stream=True, max_retries=1, health_check=True,
                      auto_eject_after_run=False, disable_thinking=True, debug_logging=False):
        return ({
            "type": "lmstudio",
            "base_url": base_url,
            "model": model,
            "connect_timeout": connect_timeout_seconds,
            "read_timeout": read_timeout_seconds,
            "stream": stream,
            "max_retries": max_retries,
            "health_check": health_check,
            "auto_eject_after_run": auto_eject_after_run,
            "disable_thinking": disable_thinking,
            "debug_logging": debug_logging,
        },)

class LMConnectLocalGGUFBackend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "C:\\models\\model.gguf"}),
            },
            "optional": {
                "mmproj_path": ("STRING", {"default": ""}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
                "n_ctx": ("INT", {"default": 8192, "min": 512, "max": 131072}),
                "n_threads": ("INT", {"default": 0}),
                "chat_format": (["auto", "chatml", "qwen2-vl", "llava-1-5", "llava-1-6", "gemma"], {"default": "auto"}),
                "keep_loaded": ("BOOLEAN", {"default": False}),
                "disable_thinking": ("BOOLEAN", {"default": True}),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LMC_BACKEND",)
    RETURN_NAMES = ("backend",)
    FUNCTION = "build_backend"
    CATEGORY = "LM Connect/Backends"

    def build_backend(self, model_path, mmproj_path="", n_gpu_layers=-1, n_ctx=8192,
                      n_threads=0, chat_format="auto", keep_loaded=False, disable_thinking=True, verbose=False):
        return ({
            "type": "local_gguf",
            "model_path": model_path,
            "mmproj_path": mmproj_path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "chat_format": chat_format,
            "keep_loaded": keep_loaded,
            "disable_thinking": disable_thinking,
            "verbose": verbose,
        },)

class LMConnectEjectLocalModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": ("*",),
            },
            "optional": {
                "only_this_backend": ("LMC_BACKEND",),
            }
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "passthrough_eject"
    CATEGORY = "LM Connect/Backends"

    def passthrough_eject(self, passthrough, only_this_backend=None):
        eject_local_model(only_this_backend)
        return (passthrough,)

class LMConnectEjectLMStudioModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": ("*",),
            },
            "optional": {
                "base_url": ("STRING", {"default": "http://localhost:1234/v1"}),
                "model": ("STRING", {"default": ""}),
                "debug_logging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "passthrough_eject"
    CATEGORY = "LM Connect/Backends"

    def passthrough_eject(self, passthrough, base_url="http://localhost:1234/v1", model="", debug_logging=False):
        result = eject_lmstudio_model(base_url, model if model else None, debug=debug_logging)
        if debug_logging and result:
            print(result)
        return (passthrough,)

class LMConnectLoadLMStudioModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": ("*",),
                "model": ("STRING", {"default": ""}),
            },
            "optional": {
                "base_url": ("STRING", {"default": "http://localhost:1234/v1"}),
                "debug_logging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "passthrough_load"
    CATEGORY = "LM Connect/Backends"

    def passthrough_load(self, passthrough, model, base_url="http://localhost:1234/v1", debug_logging=False):
        result = load_lmstudio_model(base_url, model if model else None, debug=debug_logging)
        if debug_logging and result:
            print(result)
        return (passthrough,)

