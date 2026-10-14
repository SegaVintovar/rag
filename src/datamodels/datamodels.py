from pydantic import BaseModel, Field, model_validator
import uuid


class Chunk(BaseModel):
    data_str: str

class MinimalSource(BaseModel):
    """
    MinimalSource model represents a single source of information
    """
    file_path: str
    first_character_index: int
    last_character_index: int
    data_str: str
    max_chunk_size: int
    chunks: list[Chunk]

    @model_validator(mode='after')
    def chunk_or_not_to_chunk(self) -> None:
        if len(self.data_str) > self.max_chunk_size:
            # chunk it
            print(f"{self.file_path} has to be splitted")
        else:
            self.chunks.append(Chunk(data_str=self.data_str))


#  represent an unanswered question
class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


#  represent an answered question
class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


# RagDataset model represents a dataset of RAG questions:
class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


# models represent the search results
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


# models represent the answer
class MinimalAnswer(MinimalSearchResults):
    answer: str


# StudentSearchResults models represent search results
class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


# StudentSearchResultsAndAnswer models represent search results with answers
class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int
