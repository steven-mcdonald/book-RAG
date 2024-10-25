from dataclasses import dataclass
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from get_chroma_db import get_chroma_db
# if running as a quick test in local environment you can use the path below
# from get_chroma_db import get_chroma_db


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question} in the style of {character}

Note: that each chapter starts with the chapter number for example chapter 1 starts with -1-
"""

# BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_MODEL_ID = "meta.llama3-70b-instruct-v1:0"


@dataclass
class QueryResponse:
    query_text: str
    response_text: str
    sources: List[str]


def query_rag(query_text: str, character='Jeeves') -> QueryResponse:
    db = get_chroma_db()
    # # Persist the database to disk
    # no longer need to persist
    # db.persist()

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text, character=character)
    print(f"Prompt :{prompt}")

    model = ChatBedrock(credentials_profile_name='default',
                        region_name="eu-west-2",
                        model_id=BEDROCK_MODEL_ID)
    response = model.invoke(prompt)
    response_text = response.content

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"Response: {response_text}\nSources: {sources}")

    return QueryResponse(
        query_text=query_text, response_text=response_text, sources=sources
    )


if __name__ == "__main__":
    response = query_rag(
        """How does the second chapter of 'Right Ho, Jeeves start?""", character='Bertie Wooster')

    print()
