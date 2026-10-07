from .datamodels import MinimalSource, \
    UnansweredQuestion, AnsweredQuestion, \
    RagDataset, MinimalSearchResults, \
    MinimalAnswer, StudentSearchResults, \
    StudentSearchResultsAndAnswer
import fire


class Rag():
    def index(self, max_chunk_size: int = 2000):
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
