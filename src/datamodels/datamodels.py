from pydantic import BaseModel, Field, model_validator, ConfigDict
import uuid
from enum import Enum

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, \
                                        RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


class DataType(Enum):
    MD = "md"
    PY = "py"


class Chunk(BaseModel):
    data_str: str
    first_character_index: int | None = None
    last_character_index: int | None = None


class MinimalSource(BaseModel):
    """
    MinimalSource model represents a single source of information
    """
    ###
    # Tell pydantic to accept classes it doesn't know,
    # it will only check isinstance() instead of building a schema
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ###
    file_path: str
    first_character_index: int | None = 0
    last_character_index: int | None = None
    data_str: str
    max_chunk_size: int
    chunks: list[Chunk] = []
    data_type: DataType
    splitter: RecursiveCharacterTextSplitter | CharacterTextSplitter

    @model_validator(mode='after')
    def chunk_or_not_to_chunk(self) -> None:
        if len(self.data_str) > self.max_chunk_size:
            # chunk it
            # print(f"{self.file_path} has to be splitted")
            splits = self.splitter.split_text(self.data_str)
            for ch in splits:
                fci = self.data_str.index(ch)               #  first char ids
                lci = self.data_str.index(ch[-10:]) + 10     #  last char ids
                self.chunks.append(Chunk(
                    data_str=ch,
                    first_character_index=fci,
                    last_character_index=lci
                ))
        else:
            # print(f"{self.file_path} been added to the list of chunks")
            self.chunks.append(Chunk(
                data_str=self.data_str,
                first_character_index=0,
                last_character_index=len(self.data_str)
                ))
        return self


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
