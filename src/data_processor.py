import re 
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
        data_list = []
        for file_name in os.listdir(self.data_path):
            if not file_name.endswith(".txt"):
                continue
            file_path = os.path.join(self.data_path, file_name)
            with open(file_path, "r", encoding="utf-8", errors="replace") as file_ref:
                data_list.append(
                    {
                        "file_name": file_name,
                        "page_content": file_ref.read()
                    }
                )

        return data_list

    @staticmethod
    def _decode_unicode(text: str) -> str:
        """
        Convert escaped unicode sequences to proper text.
        """
        if not isinstance(text, str):
            return text
        try:
            return text.encode("utf-8", "ignore").decode("utf-8", "ignore")
        except Exception:
            return text

    def _extract_body(self, text: str) -> str:

      if not text:
        return ""
      
      text = text.replace("\r\n", "\n").replace("\r", "\n")
      text = re.sub(r'-\n', '', text)
      text = re.sub(r'\n{3,}', '\n\n', text)

      start_patterns = [
        r"====\s*Body", r"^Body\s*$", r"^BODY\s*$",
        r"^Abstract\s*$", r"^ABSTRACT\s*$", r"^Introduction\s*$", r"^INTRODUCTION\s*$"
      ]
      end_patterns = [
          r"====\s*Back", r"^Back\s*$", r"^BACK\s*$",
          r"^References\s*$", r"^REFERENCES\s*$", r"^Bibliography\s*$",
          r"^Acknowledg", r"^Acknowledgments\s*$", r"^ACKNOWLEDGMENTS\s*$"
      ]

      start_idx = None
      for pat in start_patterns:
          m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
          if m:
              start_idx = m.end()
              break

      if start_idx is not None:
        # find end after start_idx
        end_idx = None
        for pat in end_patterns:
            m = re.search(pat, text[start_idx:], flags=re.IGNORECASE | re.MULTILINE)
            if m:
                end_idx = start_idx + m.start()
                break
        body = text[start_idx:end_idx] if end_idx else text[start_idx:]
      
      else:
         paragraphs = re.split(r'\n{2,}', text)
         paragraphs = [p.strip() for p in paragraphs if p.strip()]

         def is_metadata_para(p: str) -> bool:
            # DOI / arXiv / ISSN / PMCID / PMID
            if re.search(r'\b10\.\d{4,9}/\S+\b', p): 
                return True
            if re.search(r'\bPMCID\b|\bPMID\b', p, re.I):
                return True
            if re.search(r'ISSN[:\s]', p, re.I):
                return True
            # common metadata keywords
            if re.search(r'Correspondence:|Affiliat|Author|ORCID|E-mail:|Contact:', p, re.I):
                return True
            if re.search(r'©|license|all rights reserved|Published|Received|Accepted', p, re.I):
                return True
            
            words = p.split()
            if len(p) < 200 and len(words) <= 12 and sum(1 for w in words if w.isupper())/max(1,len(words)) > 0.6:
                return True
            return False
          
         wc = [len(p.split()) for p in paragraphs]
         good = [ (wc_i >= 40 and not is_metadata_para(p)) for p, wc_i in zip(paragraphs, wc) ]

         best_start = best_len = 0
         cur_start = cur_len = 0
         for i, g in enumerate(good):
              if g:
                  if cur_len == 0:
                      cur_start = i
                  cur_len += 1
                  if cur_len > best_len:
                      best_len = cur_len
                      best_start = cur_start
              else:
                  cur_len = 0
         if best_len > 0:
            body = "\n\n".join(paragraphs[best_start: best_start + best_len])
         else:
            # final fallback: pick the top N paragraphs by length (they likely contain body content)
            top_idxs = sorted(range(len(paragraphs)), key=lambda i: wc[i], reverse=True)[:5]
            top_idxs.sort()
            body = "\n\n".join(paragraphs[i] for i in top_idxs)
          
      body = re.sub(r'\n{2,}References[\s\S]*$', '', body, flags=re.IGNORECASE)
      body = re.sub(r'\n{2,}Bibliography[\s\S]*$', '', body, flags=re.IGNORECASE)
      body = re.sub(r'\n{2,}Acknowledg[\s\S]*$', '', body, flags=re.IGNORECASE)

      # 4) Clean junk: remove URLs/emails, collapse whitespace
      body = re.sub(r'https?://\S+', ' ', body)
      body = re.sub(r'\S+@\S+', ' ', body)
      body = re.sub(r'\s+', ' ', body).strip()

      return body

    def _preprocess(self, data: list[dict]) -> list[dict]:
        """
        Apply preprocessing steps (e.g., unicode decoding) to raw data.
        """
        cleaned_data = []
        for record in data:
            decoded_text = self._decode_unicode(record["page_content"])
            main_body = self._extract_body(decoded_text)
            cleaned_data.append(
                {
                    "file_name": record["file_name"],
                    "page_content": main_body
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