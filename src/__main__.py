# from vllm import LLM, SamplingParams
# from dotenv import load_dotenv
import os
from .datamodels.datamodels import MinimalSource, DataType
import sys
from src.rag import Rag

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


mchs = 2000

# load_dotenv()

# sp = SamplingParams()
# HF_TOKEN = os.environ["HF_TOKEN"]

# qwen = LLM(model="Qwen/Qwen3-0.6B",
#            hf_token=HF_TOKEN,
#            gpu_memory_utilization=0.65)
# tokenizer = qwen.encode

# tokenizer = qwen.get_tokenizer()
# tokenizer.encode("What is capital of the Netherlands?")

msgs = [{"role": "user", "content": "What is capital of The Netherladnds"}]

# print(qwen.chat(messages=msgs, sampling_params=sp, use_tqdm=True))

# gemini = LLM(model="gemini-3.1-flash-lite", hf_token=HF_TOKEN)

from src.rag import Rag
rag = Rag()

corpus_folder = "vllm-0.10.1"
if not os.path.isdir(corpus_folder):
    print(f"Corpus folder do not exist: {corpus_folder}", file=sys.stderr)
    exit(1)
# try:
#     files = os.listdir(corpus_folder)
# except FileNotFoundError as e:
#     print(str(e), file=sys.stderr)
#     exit(1)
md_files = []
py_files = []

import chardet
from pathlib import Path


def detect_encoding(file_path: str):
    with open(file_path, mode='rb') as file:
        detector = chardet.UniversalDetector()
        for line in file:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']


for folder, _, fls in os.walk(corpus_folder):
    for file in fls:
        file_path = str(Path(folder)) + "/" + file
        encoding = detect_encoding(file_path)
        if file.endswith(".md"):
            md_files.append((file_path, encoding, DataType.MD))
        elif file.endswith(".py"):
            py_files.append((file_path, encoding, DataType.PY))
if len(md_files) == 0 and len(py_files) == 0:
    print(f"No files found in corpus folder: {corpus_folder}", file=sys.stderr)
md_splitter = CharacterTextSplitter(chunk_size=mchs, chunk_overlap=50)
py_splitter = RecursiveCharacterTextSplitter(chunk_size=mchs, chunk_overlap=50)
all_files = md_files + py_files
# init Chunker and pass it as an argument to the MinimalSource, so it will use it
# during model vali
for f, encode, dt in all_files:
    # spltr = md_splitter if dt == DataType.MD else py_splitter
    try:
        with open(f, mode="r", encoding=encode) as source:
            tmp = source.read()
            # print(f, "\n\n\n", tmp, "\n\n\n")
            rag.data_sources.append(MinimalSource(
                file_path=f,
                first_character_index=0,
                last_character_index=len(tmp),
                data_str=tmp,
                max_chunk_size=mchs,
                data_type=dt,
                splitter=py_splitter))
    except Exception as e:
        print(f"Could not open the file {f}, error: {str(e)}")
        # exit(1)
        continue
    # chunk_n_store()


print("\n\n\nMd files: ", len(md_files), "\n\n\nPython files: ", len(py_files))
# for file in rag.data_sources:
#     print(file.file_path)

for source in rag.data_sources[:-5]:
    print("\n\n\n------------------\n\n\nNEW DATA SOURCE:\n\n\n")
    print(source.file_path)
    for ch in source.chunks:
        print(f"Chunk\n{ch.first_character_index}\n{ch.last_character_index}\n")
        print(ch.data_str)
        print()
