import os

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "system_prompts")

SYSTEM_PROMPTS = {}

def load_all_system_prompts():
    """system_prompts/ klasöründeki tüm .txt dosyalarını okuyup SYSTEM_PROMPTS dict'ine yükler."""
    SYSTEM_PROMPTS.clear()
    if not os.path.isdir(_PROMPTS_DIR):
        print(f"[LM Connect Warning] system_prompts klasörü bulunamadı: {_PROMPTS_DIR}")
        return
    for filename in os.listdir(_PROMPTS_DIR):
        if not filename.endswith(".txt"):
            continue
        key = filename[:-4]  # uzantıyı kaldır
        filepath = os.path.join(_PROMPTS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            SYSTEM_PROMPTS[key] = f.read().strip()
    if SYSTEM_PROMPTS:
        print(f"[LM Connect] {len(SYSTEM_PROMPTS)} system prompt yüklendi: {', '.join(sorted(SYSTEM_PROMPTS.keys()))}")

load_all_system_prompts()
