from datamodels.datamodels import MinimalSource, \
    UnansweredQuestion, AnsweredQuestion, \
    RagDataset, MinimalSearchResults, \
    MinimalAnswer, StudentSearchResults, \
    StudentSearchResultsAndAnswer
import fire
# from pydantic import BaseModel
import ast


class Chunk(MinimalSource):
    ...


class ChunkingStrategy():
    """
    Cause I need to chunk .py and .md,
    they need two different strategies
    """
    ...


class Chunk(MinimalSource):
    """
    This class represents one chunk
    Here I should handle chucks that are next to it, so I can handle
    an overlap for big pieces of data(tex and code)
    """
    prev_chunk: Chunk | None = None
    prev_overlap: int = 0
    next_chunk: Chunk | None = None
    next_overlap: int = 0
    ...


class Indexer():
    """
    This class is for chunking, indexing and storing data in db
    """
    # find all python files, store their pathes as a list
    # in the loop: open and read them,
    #   if len(file) > max_chunk_size: chunk(file),
    #   ast.parse(), store result in the db
    def __init__(self, max_chunk_size) -> None:
        self.max_chunk_size = max_chunk_size

    ...


class Rag():
    def index(self, max_chunk_size: int = 2000):
        myIndexer = Indexer(max_chunk_size)
        ...

    def search(self, query: str, top_k: int):
        ...

    def search_dataset(self, dataset_path, top_k, save_directory):
        ...

    def answer(self, student_search_results_path, save_directory):
        ...

    def evaluate(self, student_search_results_path, dataset_path):
        ...


if __name__ == "__main__":
    fire.Fire(Rag)
