import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp

(repo_id, model_file_name) = ("",
                            "ggml-model27-q4.gguf")

model_path = hf_hub_download(repo_id=repo_id,
                            filename=model_file_name,
                            repo_type="model")


llm_model = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=160,
    top_p=1,
    # callback_manager=callback_manager,
    n_gpu_layers=20,
    n_batch=512,
    n_ctx=8096,
    verbose=False,
    streaming=True,
    stop=["Human:","HUMAN", "###","You are a personal assistant","Hi!"]
)


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
        ("system", "Your name is  Assistant. You are a personal assistant .You help users to learn  fundamentals and techniques with detailed answers. Answer the user's question based on the context: {context}"),
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
    return response["answer"]


if __name__== '__main__':
    vector_store = create_db(get_document())
    #main_chain = create_chain(vector_store)
    history_aware_chain = history_and_retriever(vector_store)
    
    chat_history = []
    
    while True:
        user_input = input("User: ")
        response = prepare_conversation(history_aware_chain, user_input,chat_history)
        #append user input to chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        if len(chat_history) > 4:
            chat_history = chat_history[-4:]
        print(chat_history)
        print("Human:",user_input )
        print("Assistant:",response)
        print("-----------------------")