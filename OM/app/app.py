import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread

BASE_MODEL = os.environ.get("BASE_MODEL", "google/gemma-2b-it")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", None)  # set to lora-out if uploading adapter
SYSTEM_PROMPT = (
    "You are CalmMind, a gentle, encouraging mental wellness companion. "
    "Stay within wellness support: mood check-ins, breathing, grounding, gentle motivation, stress relief, sleep hygiene, journaling prompts. "
    "If a user mentions self-harm or crisis, respond with empathy and provide appropriate helpline guidance (US 988, UK 116 123, Canada 988, Australia 13 11 14) and encourage reaching local emergency services if in danger. "
    "If a user asks for unrelated topics (e.g., coding, math, politics), politely redirect: 'I'm your wellness companion, so I focus on helping you relax and feel supported.'"
)

SAFETY_KEYWORDS = [
    "kill myself", "suicide", "self harm", "end my life", "hurt myself", "can't go on", "want to die", "cutting"
]

CRISIS_RESPONSE = (
    "I'm really glad you reached out. You deserve immediate support. If you're in danger or considering self-harm, please contact a crisis line now: "
    "US & Canada: 988 (Suicide & Crisis Lifeline) | UK & ROI: Samaritans 116 123 | Australia: Lifeline 13 11 14. "
    "If elsewhere, reach local emergency services. You're not alone. We can also take a few slow breaths together if that feels okay."
)

UNRELATED_TOPICS = ["code", "program", "python", "politics", "election", "math", "calculate"]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    if ADAPTER_PATH and os.path.isdir(ADAPTER_PATH):
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        except Exception as e:
            print("Could not load adapter:", e)
    return tokenizer, model

TOKENIZER, MODEL = load_model()


def safety_filter(user_text: str) -> str | None:
    l = user_text.lower()
    for kw in SAFETY_KEYWORDS:
        if kw in l:
            return CRISIS_RESPONSE
    # Topic redirection
    if any(w in l for w in UNRELATED_TOPICS):
        return "I'm your wellness companion, so I focus on relaxation, emotional support, and healthy coping. Let's return to how you're feeling."
    return None


def format_prompt(history):
    # Combine system + chat turns
    prompt = SYSTEM_PROMPT + "\n\n"
    for user, bot in history:
        prompt += f"User: {user}\nAssistant: {bot}\n"
    return prompt


def generate_fn(message, history):
    # history: list of [user, assistant]
    crisis = safety_filter(message)
    if crisis:
        return crisis

    full_history = history + [[message, ""]]
    prompt = format_prompt(full_history) + "Assistant:"

    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)

    streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=220,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        streamer=streamer,
        eos_token_id=TOKENIZER.eos_token_id
    )
    thread = Thread(target=MODEL.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial


def mood_selector(choice):
    suggestions = {
        "😊": "Wonderful to feel some lightness. Want a mini gratitude prompt?",
        "😔": "I'm here with you. Want a gentle grounding or journaling prompt?",
        "😡": "Anger is a valid signal. Want a quick release technique (like tension + release)?",
        "😌": "Let's deepen that calm with a 1-minute breathing focus. Inhale 4, exhale 6 for 8 rounds."
    }
    return suggestions.get(choice, "How are you feeling right now?")

with gr.Blocks(title="CalmMind Wellness Companion") as demo:
    gr.Markdown("# CalmMind AI \nA lightweight mental wellness companion. Not medical advice.")
    mood = gr.Radio(["😊", "😔", "😡", "😌"], label="Mood", value=None)
    mood_output = gr.Textbox(label="Mood Support", interactive=False)
    mood.change(fn=mood_selector, inputs=mood, outputs=mood_output)

    chat = gr.ChatInterface(
        fn=generate_fn,
        examples=["I feel anxious", "Give me a breathing exercise", "I can't sleep", "I need motivation"],
        retry_btn=None,
        undo_btn=None,
        additional_inputs=[]
    )

if __name__ == "__main__":
    demo.launch()
