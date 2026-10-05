from vllm import LLM

llm = LLM(model="Qwen/Qwen3-0.6B")

msgs = [{"role": "user", "content": "What is capital of The Netherladnds"}]

llm.chat(messages=msgs, sampling_params=)

