import os

### For troubleshooting CAS RuntimeError
### (RuntimeError: Data processing error: CAS service error : Error : single flight )
### Place os.environ at the top (right after os and before any import from huggingface_hub or transformers.)
### so it takes in effect.
os.environ["OMP_NUM_THREADS"] = "1"  # silence libgomp + keep CPU sane
os.environ["HF_HUB_ENABLE_XET"] = "0"  # disable Xet path
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # use fast Rust downloader
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "info"

import requests
import gradio as gr
from openai import OpenAI
from huggingface_hub import login, snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
import threading, torch

AUDIO_MODEL = "whisper-1"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OpenAI API key not found. Set OPENAI_API_KEY in Space settings first."

if HF_TOKEN:
    login(HF_TOKEN, add_to_git_credential=True)

client = OpenAI(api_key=OPENAI_API_KEY)

### System Properties Checks:
# for k, v in os.environ.items():
#     print(k, v)


### globals (lazy init)
tokenizer = None
model = None
cache_path = "./hf-cache"
_model_lock = threading.Lock()


def ensure_model():
    global tokenizer, model

    if model is not None:
        return

    with _model_lock:
        if model is not None:
            return

        os.makedirs(cache_path, exist_ok=True)

        model_path = snapshot_download(
            repo_id=LLAMA,
            cache_dir=cache_path,
            local_dir_use_symlinks=False,
            token=os.getenv("HF_TOKEN"),
            ### Only grab the bits Transformers needs:
            ### avoid downloading original/*.pth and
            ### keeps it to the 4× - 5 GB safetensors shards only
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "model.safetensors",
                "model-*.safetensors",
                "model.safetensors.index.json",
            ],
            resume_download=True,
        )

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=bnb,
        )


# ---- Functions ----
def transcribe_audio(audio_file_path: str) -> str:
    with open(audio_file_path, "rb") as f:
        return client.audio.transcriptions.create(
            file=f, model=AUDIO_MODEL, temperature=0.1, response_format="text"
        )


def use_messages(transcript: str):
    system_message = (
        "You're an assistant that produces meeting minutes in Markdown with a "
        "summary, key discussion points, takeaways, and action items with owners."
    )
    user_prompt = (
        "Below is a meeting transcript. Write minutes in Markdown, including a summary "
        "with attendees, location and date, discussion points, takeaways, and action items."
        f"\n\nTranscript:\n{transcript}"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def stream_minutes(messages):
    ensure_model()  ### Lazy load here
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=True).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    t = threading.Thread(target=model.generate, kwargs=dict(inputs=inputs, streamer=streamer, max_new_tokens=2000))
    t.start()
    for chunk in streamer:
        yield chunk  ### raw delta chunks


def process_audio_to_text(audio_path):
    if not audio_path:
        yield "_No audio file uploaded._"
        return
    yield "**Processing…** Transcribing the audio..."
    transcript = transcribe_audio(audio_path)
    yield "**Processing..** Loading the text model and generating minutes..."
    messages = use_messages(transcript)

    header = "**Transcription complete.\n\n"
    buf = header
    for chunk in stream_minutes(messages):
        buf += chunk
        yield buf


# ---- Gradio UI ----
with gr.Blocks(title="Minutes of Meeting Generator", css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Minutes of Meeting Generator")
    audio = gr.Audio(sources=["upload"], type="filepath", label="Upload an MP3 file")
    btn = gr.Button("Transcribe", variant="primary")
    out = gr.Markdown(
        "Result might take a couple of minutes, loading an 8b model and processing multi levels of AI models for transcribing.")
    btn.click(process_audio_to_text, inputs=audio, outputs=out)
    demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)


