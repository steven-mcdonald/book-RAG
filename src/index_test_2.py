from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import Chroma
from langchain_aws import BedrockLLM

# set up document loader
loader = TextLoader("./data/right_ho_jeeves.txt", encoding="utf8")

embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id="amazon.titan-embed-text-v2:0",
        region_name="eu-west-2",
    )

index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=embeddings,
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
)


llm = BedrockLLM(
    credentials_profile_name='default',
    region_name="eu-west-2",
    model_id='meta.llama3-70b-instruct-v1:0'
)


index = index_creator.from_loaders([loader])

query = "What happened in Cannes?"
response = index.query(query, llm=llm)

index.query_with_sources(query, llm=llm)

print()

