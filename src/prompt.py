from langchain_core.prompts import PromptTemplate

BIOMED_PROMPT = """You are a scholarly assistant analyzing historical medical texts.
Use the retrieved documents to answer the user's question as completely as possible.
If the context implies but does not explicitly state a detail, you may infer it cautiously.

Context:
{context}

Question:
{question}

Answer in a clear, factual summary style."""

prompt = PromptTemplate(
    template=BIOMED_PROMPT,
    input_variables=['question', 'context']
)