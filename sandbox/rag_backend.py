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


def load_text_test():
    """Test loading data from a text file"""
    # Create a TextLoader instance to load data from a text file
    data_load = TextLoader('./data/right_ho_jeeves.txt')
    # Load and split the data into chunks
    data_test = data_load.load_and_split()
    # Print the number of chunks
    print(len(data_test))
    # Print the content of the chunk at index 2
    print(data_test[2])


def split_test(text, chunk_size=100, chunk_overlap=10):
    """Split text into chunks"""
    # Create a RecursiveCharacterTextSplitter
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Split the text into chunks
    data_split_test = data_split.split_text(text)
    # Print the resulting chunks
    print(data_split_test)


def book_index():
    """Create a vector store index for the book"""
    # Create a TextLoader instance to load text data from a file
    data_load = TextLoader('./data/right_ho_jeeves.txt')
    # Create a text splitter instance to split text into chunks
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=100)

    # Create an embedding model
    data_embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id="amazon.titan-embed-text-v2:0",
        region_name="eu-west-2",
    )

    # Create a vector store index
    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,  # Use the specified text splitter
        embedding=data_embeddings,  # Use the specified embedding model
        vectorstore_cls=FAISS  # Use FAISS as the vector store class
    )

    # Create the vector store index from the loaded text data
    db_index = data_index.from_loaders([data_load])

    return db_index


def book_llm():
    """Create a Bedrock LLM instance"""
    llm = BedrockLLM(
        credentials_profile_name='default',
        region_name="eu-west-2",
        model_id='meta.llama3-70b-instruct-v1:0'
    )
    return llm


def book_rag_response(index, question):
    """Perform RAG query on the vector store index"""
    rag_llm = book_llm()
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=question)
    book_rag_query = index.query(prompt, llm=rag_llm)
    return book_rag_query
