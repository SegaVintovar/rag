from vllm import LLM, SamplingParams
# from sentence_transformers import SentenceTransformer
# sentece_transformers will download PyTorch for CUDA/GPU

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
# import os

load_dotenv()

sp = SamplingParams()
HF_TOKEN = os.environ["HF_TOKEN"]

qwen = LLM(model="Qwen/Qwen3-0.6B", hf_token=HF_TOKEN)
tokenizer = qwen.encode

tokenizer = qwen.get_tokenizer()
tokenizer.encode("What is capital of the Netherlands?")
msgs = [{"role": "user", "content": "What is capital of The Netherladnds"}]

qwen.chat(messages=msgs, sampling_params=sp, use_tqdm=True)

# gemini = LLM(model="gemini-3.1-flash-lite", hf_token=HF_TOKEN)
