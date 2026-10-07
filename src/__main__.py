from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import os

load_dotenv()

sp = SamplingParams()
HF_TOKEN = os.environ["HF_TOKEN"]

qwen = LLM(model="Qwen/Qwen3-0.6B", hf_token=HF_TOKEN)

msgs = [{"role": "user", "content": "What is capital of The Netherladnds"}]

qwen.chat(messages=msgs, sampling_params=sp, use_tqdm=True)

gemini = LLM(model="gemini-3.1-flash-lite", hf_token=HF_TOKEN)
