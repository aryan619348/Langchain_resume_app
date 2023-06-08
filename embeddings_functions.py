from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PlaywrightURLLoader
from dotenv import load_dotenv
from langchain import FAISS
import pickle
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def pdf_embeddings(file=""):
    loader = PyPDFLoader(file)
    data = loader.load_and_split()

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=2000, chunk_overlap=200
    )
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorStore_openAI = FAISS.from_documents(docs, embeddings)

    with open("static/uploads/pdf_embeddings.pkl", "wb") as f:
        pickle.dump(vectorStore_openAI, f)



def url_embeddings(file=""):
    urls =[file]
    loader = PlaywrightURLLoader(urls=urls)
    data = loader.load_and_split()

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=2000, chunk_overlap=200
    )
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorStore_openAI = FAISS.from_documents(docs, embeddings)

    with open("static/uploads/website_embeddings.pkl", "wb") as f:
        pickle.dump(vectorStore_openAI, f)