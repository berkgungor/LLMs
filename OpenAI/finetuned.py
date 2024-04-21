
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import os
os.environ["OPENAI_API_KEY"] = ""

loader = PyPDFLoader("_RAG_3.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
db.save_local("_FAISS")
# load from local
#db = FAISS.load_local("_FAISS/", embeddings=embeddings)

llm = ChatOpenAI(
    temperature=0.8, model="ft:gpt-3.5-turbo-1106:personal:assistant:948n7SOM"
)   
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Your name is  Assistant.  Assistant is a  assistant that is created by .  Assistant helps users to learn  fundamentals and techniques, recommends ideal  exercises to improve their  skills. If you do not know the answer, do not make up an answer, say that you do not know the answer. If the user asks you to open a game like for example PuttPong, Maze or TicTacToe, you can do this by adding brackets at the beginning and the end to the name of the game in capitals to the end of the answer, for example [PUTTPONG]"),
    ("user", "{input}")
])

chain = prompt | llm | output_parser

chain.invoke("What is the best way to improve my  skills?")