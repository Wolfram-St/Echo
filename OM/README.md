# CalmMind AI – Mental Wellness Companion (Hackathon-Ready)

Lightweight, open-source mental wellness chat companion fine-tuned (via LoRA) on a small curated dataset. Runs on free Colab (T4) for training and Hugging Face Spaces (Gradio) for deployment.

> Disclaimer: This tool is not medical advice or a replacement for professional help. In crisis, contact a qualified professional or emergency services.

## Features
- Base model: `google/gemma-2b-it` (instruction-tuned, small, permissive)
- Parameter-efficient QLoRA fine-tuning (~45–60 min on 200 examples)
- Safety guardrails: crisis keyword detection, topic redirection
- Chat UI with mood selector (emoji) + streaming responses
- Dataset template covering: mood check-ins, breathing, grounding, yoga, meditation, affirmations, journaling prompts, emergency guidance

## Repo Structure
```
├─ data/
│  └─ wellness_dataset_sample.jsonl   # Seed examples (expand to 100–200)
├─ scripts/
│  └─ train_lora.py                   # QLoRA fine-tuning script
├─ app/
│  └─ app.py                          # Gradio chat app
├─ requirements.txt
└─ README.md
```

## Dataset Format
JSONL records with keys: `instruction`, `output`.
Extend `data/wellness_dataset_sample.jsonl` to ~200 examples for better coverage.

## Quick Start (Local)
1. Create virtual env and install deps:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
2. (Optional) Add more JSONL examples.
3. Train LoRA adapter:
```bash
python scripts/train_lora.py
```
4. Launch app (base model only or with adapter):
```bash
# If you trained an adapter:
set ADAPTER_PATH=lora-out  # (Windows PowerShell: $Env:ADAPTER_PATH="lora-out")
python app/app.py
```
Open the printed local URL.

## Colab Fine-Tuning (QLoRA)
Minimal Colab cell sequence:
```python
!pip install -q transformers accelerate peft bitsandbytes datasets sentencepiece protobuf<5
from google.colab import drive; drive.mount('/content/drive')  # optional persistence
!python scripts/train_lora.py --quiet
```
Upload/clone repo, ensure dataset present, run training. Adapter saved to `lora-out`.

## Upload to Hugging Face Hub
```python
from huggingface_hub import login
login()
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained('google/gemma-2b-it')
# push adapter directory directly:
!huggingface-cli upload <your-username>/<repo-name> lora-out/ lora-out/ --include '*.bin'
```
Or merge adapter (optional) then push full model (larger upload).

## Deploy on Hugging Face Spaces
Create a Gradio Space. Place `app.py` (rename to `app.py` at root or set Space SDK) + `requirements.txt`. Set Space hardware to CPU or T4 small if available.

## Safety Layer
- Keyword crisis detection triggers helpline message.
- Topic redirect for unrelated asks (programming, politics, etc.).
Enhance by adding a lightweight moderation model or regex expansions.

## Roadmap Enhancements
- Add journaling prompt randomizer button
- Add optional background ambience (looped audio)
- Store no conversation logs (privacy) or add ephemeral session memory only
- Add analytics toggle (disabled by default)

## Presentation Tips
1. Problem: Accessible emotional support & coping micro-interventions.
2. Solution: CalmMind AI – guided breathing, grounding, gentle motivation.
3. Differentiator: Runs on free infra + custom fine-tuned dataset.
4. Demo Flow: Mood select -> Anxiety input -> Breathing guidance -> Short meditation.
5. Ethics: Clear disclaimer, crisis escalation, no diagnostic claims.

## Disclaimer
Not a medical device. Encourage professional help for persistent distress. In emergencies, contact local emergency services immediately.

---
Feel free to expand and iterate quickly during the hackathon. Good luck!
