import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import re

os.environ["OPENAI_API_KEY"] = ""
job_id = 'ftjob-Wg2imnR2U6bjgfqlYky4UXhN'

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
finetuned_job = client.fine_tuning.jobs.retrieve(job_id)
finetuned_model = finetuned_job.fine_tuned_model

llm_model = ChatOpenAI(model_name=finetuned_model, temperature=0.4 )

#llm_model = llm_model.bind(stop=[">>"])

def get_document():
    loader = PyPDFLoader("../_RAG_3.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    return docs

def create_db(docs):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embedding = embeddings)
    return db


def history_and_retriever(vector_store):
    
    #create the main chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Your name is  Assistant. You are a personal assistant . 
             You help users to learn  fundamentals and techniques with detailed answers. 
              helps people to practice  with the help of a projector and an indoor green platform so you might be asked to project a  practice on green. 
             If you do not know the answer, do not make up an answer, say that you do not know the answer. 
             If the user asks you to open a game like for example PuttPong, Maze or TicTacToe, you can do this by adding brackets at the beginning and the end to the name of the game in capitals to the end of the answer, for example [PUTTPONG].
             Answer within 3 sentences to the user's question based on the context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ])
    chain = create_stuff_documents_chain(llm = llm_model, prompt = prompt)
    
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    #combine chat history and latest user input
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    # Generate history aware retriever
    history_aware_retriever = create_history_aware_retriever(llm=llm_model, retriever = retriever, prompt= retriever_prompt)
    
    retrieval_chain = create_retrieval_chain(history_aware_retriever,chain)
    
    return retrieval_chain


def prepare_conversation(main_chain,question,chat_history):
    response = main_chain.invoke({
    "input" : question,
    "chat_history" : chat_history
})
    print(response)
    return response["answer"]


def extract_exercise_type(answer):
    pattern_after = r'<<(.*)'
    pattern_before = r'^(.*?)<'
    main_text = re.search(pattern_before, answer)
    exercise = re.search(pattern_after, answer)
    answer = main_text.group(1)
    return answer, exercise



if __name__== '__main__':
    vector_store = create_db(get_document())
    #main_chain = create_chain(vector_store)
    history_aware_chain = history_and_retriever(vector_store)
    
    chat_history = []
    
    while True:
        user_input = input("User: ")
        response = prepare_conversation(history_aware_chain, user_input,chat_history)
        #append user input to chat history
        #answer, exercise = extract_exercise_type(response)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        if len(chat_history) > 4:
            chat_history = chat_history[-4:]
        print(chat_history)
        print("Human:",user_input )
        print("Assistant:",response)
        #print("Exercise Type:",exercise)
        print("-----------------------")