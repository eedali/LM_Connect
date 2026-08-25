from .nodes import LMConnectSimplePrompt, LMConnectPromptWithSystem, LMConnectVision
from .nodes_h3 import LMConnectH3Prompt, LMConnectH3PromptFullReference, LMConnectH3PersonSwap, LMConnectExtraSystemPrompt, LMConnectH3ImageToVideoPrompt
from .nodes_backend import LMConnectLMStudioBackend, LMConnectLocalGGUFBackend, LMConnectEjectLocalModel, LMConnectEjectLMStudioModel, LMConnectLoadLMStudioModel

NODE_CLASS_MAPPINGS = {
    "LMConnectSimplePrompt": LMConnectSimplePrompt,
    "LMConnectPromptWithSystem": LMConnectPromptWithSystem,
    "LMConnectVision": LMConnectVision,
    "LMConnectH3Prompt": LMConnectH3Prompt,
    "LMConnectH3PromptFullReference": LMConnectH3PromptFullReference,
    "LMConnectH3PersonSwap": LMConnectH3PersonSwap,
    "LMConnectH3ImageToVideoPrompt": LMConnectH3ImageToVideoPrompt,
    "LMConnectExtraSystemPrompt": LMConnectExtraSystemPrompt,
    "LMConnectLMStudioBackend": LMConnectLMStudioBackend,
    "LMConnectLocalGGUFBackend": LMConnectLocalGGUFBackend,
    "LMConnectEjectLocalModel": LMConnectEjectLocalModel,
    "LMConnectEjectLMStudioModel": LMConnectEjectLMStudioModel,
    "LMConnectLoadLMStudioModel": LMConnectLoadLMStudioModel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LMConnectSimplePrompt": "LM Connect: Simple Prompt",
    "LMConnectPromptWithSystem": "LM Connect: Prompt + System Prompt",
    "LMConnectVision": "LM Connect: Vision (up to 5 images)",
    "LMConnectH3Prompt": "LM Connect: H3 Prompt (T2VA/I2VA/FL2VA/L2VA)",
    "LMConnectH3PromptFullReference": "LM Connect: H3 Prompt (Full Reference)",
    "LMConnectH3PersonSwap": "LM Connect: H3 Person Swap",
    "LMConnectH3ImageToVideoPrompt": "LM Connect: H3 Image to Video Prompt",
    "LMConnectExtraSystemPrompt": "LM Connect: Extra System Prompt",
    "LMConnectLMStudioBackend": "LM Connect: LM Studio Backend",
    "LMConnectLocalGGUFBackend": "LM Connect: Local GGUF Backend",
    "LMConnectEjectLocalModel": "LM Connect: Eject Local Model",
    "LMConnectEjectLMStudioModel": "LM Connect: Eject LM Studio Model",
    "LMConnectLoadLMStudioModel": "LM Connect: Load LM Studio Model"
}

WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
