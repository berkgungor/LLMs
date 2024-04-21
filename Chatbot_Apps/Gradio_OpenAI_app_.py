import gradio as gr
import csv
import time 
import os
from openai import OpenAI
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.chains import LLMChain
from langchain.memory import VectorStoreRetrieverMemory


def prepare_model():
    os.environ["OPENAI_API_KEY"] = ""
    job_id = 'ftjob-hjLV1XIM6ccPjzx6zTpAYw5z'

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    finetuned_job = client.fine_tuning.jobs.retrieve(job_id)
    finetuned_model = finetuned_job.fine_tuned_model
    return finetuned_model

def prepare_memory():
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""Your name is  Assistant. You are a personal assistant . Recommend ideal  exercises to help users to improve their  skills.
    You help users to learn  fundamentals and techniques with detailed answers. You also enlight users about  products. Give answers with 2 or 3 sentences, not longer.
    """
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  
            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),  
        ]
    )

    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=2)
    return prompt,memory
    

def prepare_rag():
    loader = PyPDFLoader("_RAG_3.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("_FAISS")
    return db


def record_data(user_data, assistant_data,feedback, response_time ):
    csv_file = 'chatbot_data_record_gradio.csv'
    column_names = ['User', 'Assistant', 'ResponseTime', 'Feedback']

    try:
        with open(csv_file, 'r', newline='') as file:
            pass
    except FileNotFoundError:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_data, assistant_data, response_time, feedback])

def create_chain():
    model = prepare_model()
    prompt,memory = prepare_memory()
    db = prepare_rag()
    llm = ChatOpenAI(model_name=model, temperature=0)
    retriever = db.as_retriever()
    
    chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
    )
    
    return chat_llm_chain


def main():
    chat_llm_chain = create_chain()
    
    def generate_text(prompt):
        start_time = time.time()
        response = chat_llm_chain.predict(human_input=prompt)
        response_time = time.time() - start_time
        feedback = ""
        record_data(prompt, response, feedback, response_time)
        return response

    
    with gr.Blocks() as demo:
        gr.HTML(
            """
            <h1 style="font-weight: 900; margin-bottom: 7px;">
            Golf Assistant v2.0
            </h1>
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                user_data = gr.Textbox(
                    lines=4,
                    label="Welcome to  Golf Assistant.",
                    interactive=True,
                    value="What is Ladder Drill?",
                )
                generate_button = gr.Button("Ask  Assistant")
                

            with gr.Column(scale=2):
                response = gr.Textbox(
                    lines=4, label="Please wait few seconds for answer...", interactive=False
                )
                
            # add textbox for feedback
            with gr.Column(scale=1):
                feedback = gr.Textbox(
                    lines=4,
                    label="Give Feedback / Fix the Answer",
                    interactive=True,
                    value="",
                )
                feedback_button = gr.Button("Give Feedback")

        gr.Markdown("From [](https://www..com/)")

        generate_button.click(
            fn=generate_text,
            inputs=[
                user_data,
            ],
            outputs=response,
        )
        
        feedback_button.click(
            fn=record_data,
            inputs=[
                user_data,
                response,
                feedback,
            ],
            outputs=None,
        )

    demo.launch(share=True)
   
        
if __name__ == "__main__":
    main()