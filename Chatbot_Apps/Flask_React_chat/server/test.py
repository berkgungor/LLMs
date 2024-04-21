import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
job_id = 'ftjob-Wg2imnR2U6bjgfqlYky4UXhN'
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
finetuned_job = client.fine_tuning.jobs.retrieve(job_id)
llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=200)

def normal_llm(assistantType):
    
    if assistantType == "marketing":
        system_prompt = """You are a marketing assistant. You help users to develop their marketing strategies. The main products that you try to do marketing about is  simulator called .
            if you do not know the answer, do not make up an answer, say that you do not know the answer. Question: {input}
            Answer:"""
    elif assistantType == "sales":
        system_prompt = """You are a sales assistant. You help other sales agents to develop their sales strategies. The main products that you try to sell is  simulator called .
            if you do not know the answer, do not make up an answer, say that you do not know the answer. Question: {input}
            Answer:"""
    elif assistantType == "software":
        system_prompt = """You are a software assistant. You help users to write codes and fix their bugs.
            if you do not know the answer, do not make up an answer, say that you do not know the answer. Question: {input}
            Answer:"""

    normal_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    llm_chain = LLMChain(prompt=normal_prompt, llm=llm_model)

    return llm_chain

chain = normal_llm("marketing")
response= chain.invoke({"chat_history": [], "input": "where is the capital of singapur"})
print(response["text"])