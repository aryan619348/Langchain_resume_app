#will list all the tools and functions here itself

from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import pickle
from dotenv import load_dotenv
import os
#from langchain.tools import DuckDuckGoSearchTool
from langchain.agents import Tool
from langchain.tools import BaseTool
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo')





def resume_question(input=""):
    with open("static/uploads/pdf_embeddings.pkl", "rb") as f:
        docsearch = pickle.load(f)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    docs = docsearch.similarity_search(input)
    answer=chain.run(input_documents=docs, question=input)
    return answer



# #search = DuckDuckGoSearchTool()

def jobsite_question(input="",jobsite=""):
    with open("static/uploads/website_embeddings.pkl", "rb") as f:
        jobsite = pickle.load(f)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    docs = jobsite.similarity_search(input)
    answer=chain.run(input_documents=docs, question=input)
    return answer
