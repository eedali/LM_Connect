# LM Connect for ComfyUI

A comprehensive suite of custom nodes for ComfyUI designed to seamlessly connect your workflows to local LLM models (Language Models), either via LM Studio's API or directly within ComfyUI using local GGUF models.

## Key Features

- **Multi-Backend Support**: Connect to external LLM servers like LM Studio via REST API, or run `.gguf` models directly inside the ComfyUI process using `llama-cpp-python`.
- **Smart VRAM Management**: Passthrough Eject nodes allow you to automatically unload models from VRAM after generating a text prompt, freeing up your GPU entirely for heavy video/image generation tasks.
- **Advanced API Handling**: Robust HTTP implementation with streaming support to prevent ComfyUI from freezing or timing out during long prompt generation sessions.
- **Vision Model Support**: Dynamic image inputs (up to 5 images) and automatic contact sheet generation for video references to be used with Vision-Language Models.
- **MiniMax H3 Prompt Generation**: Specialized nodes crafted to strictly follow the official MiniMax H3 Prompt Writing Guides (T2VA, I2VA, FL2VA, L2VA) to create highly detailed, structured video prompts.
- **Compact Guide Mode**: Reduces system prompt token usage by ~75% while preserving all formatting rules for H3 prompts.
- **Thinking Model Support**: Automatically disables "thinking/reasoning" mode on models (like Qwen3) to prevent empty outputs when using limited context/token sizes, and provides helpful warnings if a model stalls in reasoning.

## Installation

1. Navigate to your ComfyUI `custom_nodes` folder.
2. Clone this repository (or copy the `LM_Connect` folder here).
3. Install the basic requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional but Highly Recommended for Local GGUF)** Install `llama-cpp-python` with CUDA support to run models directly inside ComfyUI:
   - You must install a wheel that matches your CUDA version. For example, for CUDA 12.1:
   ```bash
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   ```
   *(Change `cu121` to your installed CUDA version, e.g., `cu118`, `cu122`).*
5. Restart ComfyUI.

## The Backend System

All text and prompt generation nodes now have an **optional** `backend` input.
- **Legacy Mode**: If left disconnected, the node will fall back to using basic LM Studio fields (`base_url`, `model`, etc.) provided directly on the node.
- **Advanced Backend**: If you connect a Backend node, you gain advanced control over streaming, timeouts, memory, and model behavior.

### 1. LM Studio Advanced Backend (`LMConnectLMStudioBackend`)
Ideal if you prefer running models via the LM Studio application in the background.
- **Streaming**: Prevents ComfyUI UI lockups and timeouts during long prompt generation.
- **Auto Eject**: Can trigger LM Studio's Native API to unload the model immediately after finishing the prompt.
- **`disable_thinking`**: Enabled by default. Sends `chat_template_kwargs: {enable_thinking: false}` to prevent models from exhausting their token limit on internal reasoning (common in models like DeepSeek-R1 or Qwen3).
- **`debug_logging`**: Prints raw API responses and eject results to the ComfyUI console for troubleshooting.

### 2. Local GGUF Backend (`LMConnectLocalGGUFBackend`)
Run `.gguf` models directly *inside* ComfyUI's Python process using `llama-cpp-python`.
- **Why use this?** No need to run a separate LM Studio process. It integrates perfectly with ComfyUI's execution queue and can automatically free VRAM when done.
- **`mmproj_path`**: For Vision models (like Qwen2-VL or LLaVA), you must provide the matching multimodal projector (`mmproj-*.gguf`) file alongside the main model.
- **`n_ctx`**: Defines the context window size. Increase this (e.g., to 8192 or 16384) if your H3 Prompt generation fails due to context limits.

## Ejecting Models (VRAM Management)

To ensure your LLM does not hoard VRAM while your heavy Video/Image diffusion models are running, LM Connect provides **Passthrough Eject Nodes**:

- **LM Connect: Eject Local Model**
- **LM Connect: Eject LM Studio Model**
- **LM Connect: Load LM Studio Model** (To proactively load models before they are needed).

**Suggested VRAM-Optimized Workflow:**
1. Generate your prompt using a generation node (e.g., `LM Connect: H3 Prompt (Full Reference)`) connected to a Local GGUF Backend.
2. Connect the `h3_prompt` output string into the `passthrough` input of the **Eject Local Model** node.
3. Connect the `passthrough` output of the Eject node into your Text Encoding / Video Generation node (e.g., HunyuanVideo or MiniMax).
*Result:* ComfyUI will wait for the text prompt to be generated, pass it through the Eject node (which forcefully clears the LLM from VRAM), and *then* begin the heavy video generation with 100% free VRAM!

> **Alternative Eject Tip:** If the LM Studio eject API fails for any reason, you can achieve the same result manually in LM Studio by going to the Developer tab and setting "Auto-Unload" / "Idle TTL" to a short duration (e.g., 30-60 seconds) for your models.

## Context Size & Token Management

When using the advanced H3 prompt nodes with vision models, the combined length of the official guide text and base64 images can easily exceed the model's context window. LM Connect provides several tools to manage this:

- **`guide_mode`** (dropdown: `full` / `compact`): Default is `compact`. The compact guides are ~75% shorter while preserving all core formatting rules, labels, and prohibitions required by MiniMax.
- **`max_image_dimension`** (INT, default 768): Automatically scales down images before encoding them to base64. This drastically reduces vision token consumption.
- **`assume_context_size`** (INT, default 8192): If the estimated token count exceeds this value, a warning is prepended to the generated output suggesting corrective actions.

> **Tip:** When loading a model in LM Studio, increasing the "Context Length" in the right panel (to 16384 or 32768, VRAM permitting) permanently resolves most context overflow errors.

## Available Nodes Reference

### 🔌 Backends (`LM Connect/Backends`)
- **LM Connect: LM Studio Backend**
- **LM Connect: Local GGUF Backend**
- **LM Connect: Eject Local Model**
- **LM Connect: Eject LM Studio Model**
- **LM Connect: Load LM Studio Model**

### ✍️ General Prompting (`LM Connect`)
- **LM Connect: Simple Prompt** - Basic text generation.
- **LM Connect: Prompt + System Prompt** - Text generation with a dedicated system prompt.
- **LM Connect: Vision (up to 5 images)** - Process up to 5 images in a single LLM request.

### 🎬 MiniMax H3 Generation (`LM Connect/MiniMax H3`)
- **LM Connect: H3 Prompt (T2VA/I2VA/FL2VA/L2VA)** - Specialized structured prompt generation for MiniMax models.
- **LM Connect: H3 Prompt (Full Reference)** - Similar to the above, but supports full image/video references (via automatic contact sheets) and audio descriptors.
- **LM Connect: Extra System Prompt** - Allows injecting additional custom rules into the H3 system prompt.
