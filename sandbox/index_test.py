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

# load book
documents = loader.load()

# set up text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# split text
texts = text_splitter.split_documents(documents)

embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id="amazon.titan-embed-text-v2:0",
        region_name="eu-west-2",
    )

db = Chroma.from_documents(documents=texts, embedding=embeddings)

llm = BedrockLLM(
        credentials_profile_name='default',
        region_name="eu-west-2",
        model_id='meta.llama3-70b-instruct-v1:0'
    )

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever)

# query = "Who is 'Right Ho, Jeeves' dedicated to?"
query = "What happened in Cannes?"
response = qa.run(query)


index.query_with_sources(query)

print()
