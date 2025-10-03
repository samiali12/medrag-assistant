import os
from src.constant import BASE_DIR
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = os.path.join(BASE_DIR, "data", "pmc")


class DataProcessor:
    """
    Handles loading, cleaning, and chunking of text files 
    from the PubMed Central (PMC) dataset.
    """

    def __init__(self, data_path: str = DATA_DIR):
        self.data_path = data_path

    def _load_files(self) -> list[dict]:
        """
        Load raw text files from the dataset directory.
        Returns a list of dictionaries with file name and raw content.
        """
        count = 0
        data_list = []
        for file_name in os.listdir(self.data_path):
            if not file_name.endswith(".txt"):
                continue
            file_path = os.path.join(self.data_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file_ref:
                data_list.append(
                    {
                        "file_name": file_name,
                        "page_content": file_ref.read()
                    }
                )
            if count >= 3000:
                break
            count += 1
            
        return data_list

    @staticmethod
    def _decode_unicode(text: str) -> str:
        """
        Convert escaped unicode sequences to proper text.
        """
        if not isinstance(text, str):
            return text
        try:
            return text.encode("utf-8").decode("unicode-escape")
        except Exception:
            return text

    def _preprocess(self, data: list[dict]) -> list[dict]:
        """
        Apply preprocessing steps (e.g., unicode decoding) to raw data.
        """
        cleaned_data = []
        for record in data:
            decoded_text = self._decode_unicode(record["page_content"])
            cleaned_data.append(
                {
                    "file_name": record["file_name"],
                    "page_content": decoded_text
                }
            )
        return cleaned_data

    def load_documents(self) -> list[Document]:
        """
        Load and preprocess text files, converting them into 
        LangChain Document objects.
        """
        raw_data = self._load_files()
        cleaned_data = self._preprocess(raw_data)

        return [
            Document(
                page_content=item["page_content"],
                metadata={"source": item["file_name"]}
            )
            for item in cleaned_data
        ]

    @staticmethod
    def chunk_documents(documents: list[Document],
                        chunk_size: int = 1000,
                        chunk_overlap: int = 200) -> list[Document]:
        """
        Split documents into smaller chunks for embedding and retrieval.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return splitter.split_documents(documents)

    def build(self) -> tuple[list[Document], list[Document]]:
        """
        End-to-end pipeline:
        - Load documents
        - Chunk them
        Returns (chunks, original documents).
        """
        documents = self.load_documents()
        chunks = self.chunk_documents(documents)
        return chunks, documents
