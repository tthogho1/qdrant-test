import bs4
import configparser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

from openai import OpenAI

config = configparser.ConfigParser()
config.read('config.ini')
apikey = config['apikey']['openapikey']
    
embeddings = OpenAIEmbeddings(
    openai_api_key=apikey
)

qdrantkey = config['apikey']['qdrantkey']
qdranturl = config['apikey']['qdranturl']

vectorstore = Qdrant(
    collection_name="my_collection",
    client=QdrantClient(
        api_key=qdrantkey,
        url=qdranturl        
    ),
    embeddings=embeddings
)

dbclient = QdrantClient(
        api_key=qdrantkey,
        url=qdranturl        
)

isCollectionExist = dbclient.collection_exists(
    collection_name="my_collection"
)

if not isCollectionExist:
    dbclient.create_collection(
        collection_name="my_collection",
        vectors_config={"size": 1536, "distance": "Cosine"}
    )
    
    urls =[
        "https://en.wikipedia.org/wiki/List_of_world_heavyweight_boxing_champions",
        "https://en.wikipedia.org/wiki/List_of_boxing_quadruple_champions",
        "https://en.wikipedia.org/wiki/Category:World_boxing_champions",
        "https://en.wikipedia.org/wiki/List_of_current_world_boxing_champions",
        "https://en.wikipedia.org/wiki/List_of_undisputed_world_boxing_champions" 
    ]

    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore.add_documents(splits)


retriever = vectorstore.as_retriever()

client = OpenAI(
    api_key=apikey
)

def openai_llm(question, context):
    messages = [
        {"role": "system", "content": question},
        {"role": "user", "content": context},
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return openai_llm(question, formatted_context)

result = rag_chain("who is the current world heavyweight boxing champion?")
print(result)
