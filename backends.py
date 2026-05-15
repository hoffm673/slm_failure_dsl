"""
Model backends.

You want the pipelines runnable without a GPU/model loaded so you can develop
the harness on CPU. So we provide:
  - StubModel: deterministic-ish, no real inference, useful for unit tests
  - TransformersModel: wraps a HuggingFace model, the real backend
"""

from __future__ import annotations

import hashlib
import random
from typing import Protocol


class ModelBackend(Protocol):
    name: str
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str: ...


class StubModel:
    def __init__(self, name: str = "stub", seed: int = 0, failure_rate: float = 0.15):
        self.name = name
        self._rng = random.Random(seed)
        self.failure_rate = failure_rate

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        rng = random.Random(h + self._rng.randint(0, 10**9))
        roll = rng.random()
        if roll < self.failure_rate * 0.3:
            return "I'm sorry, but I can't help with that."
        if roll < self.failure_rate * 0.6:
            return "BANANA " + "lorem ipsum " * 5
        if roll < self.failure_rate:
            return "{ not valid json"
        if "json" in prompt.lower() or "{" in prompt:
            return '{"summary": "stub output", "entities": ["Alice", "Bob"], "sentiment": "neutral"}'
        return "Stub output for prompt of length " + str(len(prompt))


class TransformersModel:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16"):
        self.name = model_id
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=self.device,
        )
        self._model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        self._ensure_loaded()
        import torch

        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            templated = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(templated, return_tensors="pt").to(self.device)
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        generated = output[0, input_ids.shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)


MODEL_PAIRS = {
    "qwen": ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"),
    "smollm": ("HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
}