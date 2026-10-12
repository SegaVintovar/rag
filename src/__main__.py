from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import os
from rag.datamodels.datamodels import MinimalSource


from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

sp = SamplingParams()
HF_TOKEN = os.environ["HF_TOKEN"]

qwen = LLM(model="Qwen/Qwen3-0.6B", hf_token=HF_TOKEN)
tokenizer = qwen.encode

tokenizer = qwen.get_tokenizer()
tokenizer.encode("What is capital of the Netherlands?")

msgs = [{"role": "user", "content": "What is capital of The Netherladnds"}]

print(qwen.chat(messages=msgs, sampling_params=sp, use_tqdm=True))

# gemini = LLM(model="gemini-3.1-flash-lite", hf_token=HF_TOKEN)

from rag import Rag
rag = Rag()

corpus_folder = "vllm-0.10.1"
for filename in os.listdir(corpus_folder):
    if filename.endswith(".md"):
        rag.data_sources.append(
            MinimalSource(file_path=(os.path + "/" + filename)))

for file in rag.data_sources:
    print(file.file_path)
