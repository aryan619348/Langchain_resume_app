from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import pickle
from dotenv import load_dotenv
import os
from langchain.agents import Tool
from langchain.tools import BaseTool
import json

from custom_tools import resume_question, jobsite_question

from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


llm = ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo')

resume_tool = Tool(
    name='resume_question',
    func= resume_question,
    description="Useful for when you need to answer questions about the resume. input is the question"
)

job_tool = Tool(
    name='job_question',
    func= jobsite_question,
    description="Useful for when you need to answer questions about the job.input is the question"
)

tools =[resume_tool,job_tool]

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)

conversational_agent = initialize_agent(
    agent='zero-shot-react-description',
    tools=tools,
    llm=llm,
    max_iterations=3,
    early_stopping_method='generate',
    memory=memory
)

def process_answer(question=""):
    answer = conversational_agent(question)
    output = answer["output"]
    return output