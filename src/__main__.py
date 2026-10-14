# from vllm import LLM, SamplingParams
# from dotenv import load_dotenv
import os
from .datamodels.datamodels import MinimalSource
import sys

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


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
for folder, _, fls in os.walk(corpus_folder):
    for file in fls:
        if file.endswith(".md"):
            md_files.append(folder + "/" + file)
        elif file.endswith(".py"):
            py_files.append(folder + "/" + file)
if len(md_files) == 0 and len(py_files) == 0:
    print(f"No files found in corpus folder: {corpus_folder}", file=sys.stderr)

all_files = md_files + py_files
for f in all_files:
    try:
        with open(f, mode="r") as source:
            tmp = source.read()
            rag.data_sources.append(MinimalSource(
                file_path=f,
                first_character_index=0,
                last_character_index=len(tmp),
                data_str=tmp))
    except Exception as e:
        print(f"Could not open the file {f}, error: {str(e)}")
        exit(1)
    # chunk_n_store()
    # if len(tmp) > 2000:
        
        # print(file)
    # if filename.endswith(".md"):
    #     rag.data_sources.append(
    #         MinimalSource(file_path=(os.path + "/" + filename)))
print("Md files: ", len(md_files), "\n\n\nPython files: ", len(py_files))
# for file in rag.data_sources:
#     print(file.file_path)
