"""Utilities for loading models and generating answers."""
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = PROJECT_ROOT / "model" / "DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "models" / "lora-full"
BYTE_DECODER = None


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def read_adapter_base_model(adapter_dir: str | Path) -> str | None:
    config_path = resolve_path(adapter_dir) / "adapter_config.json"
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config.get("base_model_name_or_path")


def load_model(
    base_model: str | Path | None = None,
    adapter_dir: str | Path | None = None,
    device_map: str = "auto",
):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Generation requires torch and transformers.") from error

    selected_base = str(resolve_path(base_model or DEFAULT_BASE_MODEL))
    if adapter_dir:
        selected_base = read_adapter_base_model(adapter_dir) or selected_base

    tokenizer = AutoTokenizer.from_pretrained(selected_base, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        selected_base,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if adapter_dir:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("LoRA adapter loading requires peft.") from error
        model = PeftModel.from_pretrained(model, str(resolve_path(adapter_dir)))

    model.eval()
    return tokenizer, model


def render_prompt(tokenizer, prompt: str | None = None, messages: list[dict] | None = None) -> str:
    if messages and getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Match LLaMA-Factory deepseekr1 with enable_thinking=false: keep the
        # empty thinking block in the prompt so generation starts at the answer.
        if rendered.endswith("<think>\n"):
            rendered += "\n</think>\n\n"
        return rendered
    if prompt is None:
        raise ValueError("Either prompt or messages must be provided.")
    return prompt


def generate_answer(
    tokenizer,
    model,
    prompt: str | None = None,
    messages: list[dict] | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.08,
    no_repeat_ngram_size: int = 4,
) -> str:
    import torch

    rendered_prompt = render_prompt(tokenizer, prompt=prompt, messages=messages)
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    do_sample = temperature > 0
    if not do_sample:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "renormalize_logits": True,
    }
    if no_repeat_ngram_size > 0:
        generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return clean_generation_text(text)


def clean_generation_text(text: str) -> str:
    decoded = decode_byte_level_text(text)
    return decoded.strip()


def bytes_to_unicode() -> dict[int, str]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    chars = visible[:]
    next_code = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            chars.append(256 + next_code)
            next_code += 1
    return dict(zip(visible, [chr(char) for char in chars], strict=True))


def decode_byte_level_text(text: str) -> str:
    global BYTE_DECODER
    if BYTE_DECODER is None:
        BYTE_DECODER = {char: byte for byte, char in bytes_to_unicode().items()}

    raw = bytearray()
    for char in text:
        if char in BYTE_DECODER:
            raw.append(BYTE_DECODER[char])
        else:
            raw.extend(char.encode("utf-8"))
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return (
            text.replace("\u0120", " ")
            .replace("\u010a", "\n")
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
        )


def row_question(row: dict) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if instruction and input_text:
        return f"{instruction}\n{input_text}"
    return instruction or input_text
