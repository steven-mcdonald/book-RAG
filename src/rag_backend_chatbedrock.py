from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_aws import BedrockLLM
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Answer the question based only on the context provided: {question}
"""


def load_test():

    data_load = TextLoader('./data/right_ho_jeeves.txt')
    data_test = data_load.load_and_split()
    print(len(data_test))
    print(data_test[2])


def split_test(text, chunk_size=100, chunk_overlap=10):
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    data_split_test = data_split.split_text(text)
    print(data_split_test)


def book_index():

    data_load = TextLoader('./data/right_ho_jeeves.txt')
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=100)

    data_embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id="amazon.titan-embed-text-v2:0",
        region_name="eu-west-2",
    )

    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS
    )

    db_index = data_index.from_loaders([data_load])

    return db_index


def book_llm():
    llm = BedrockLLM(
        credentials_profile_name='default',
        region_name="eu-west-2",
        model_id='meta.llama3-70b-instruct-v1:0'
    )
    return llm


def book_rag_response(index, question):

    rag_llm = book_llm()
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=question)
    book_rag_query = index.query(prompt, llm=rag_llm)
    return book_rag_query
