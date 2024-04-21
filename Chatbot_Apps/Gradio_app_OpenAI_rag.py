from langchain_core.prompts import format_document  
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain, ConversationChain,ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import gradio as gr
import csv
import time
import PyPDF2
import os
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage


def prepare_model():
    os.environ["OPENAI_API_KEY"] = ""
    job_id = 'ftjob-hjLV1XIM6ccPjzx6zTpAYw5z'

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    finetuned_job = client.fine_tuning.jobs.retrieve(job_id)
    finetuned_model = finetuned_job.fine_tuned_model
    return finetuned_model

def prepare_rag():
    loader = PyPDFLoader("RAG_3.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("_FAISS")
    return db

def record_data(user_data, assistant_data,feedback, response_time ):
    csv_file = 'chatbot_data_record_gradio_v2.csv'
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

def main():

    template = """
    Your name is  Assistant. You are a personal assistant . Recommend ideal  exercises to help users to improve their  skills.
    You help users to learn  fundamentals and techniques with detailed answers. You also enlight users about  products. Give answers with 2 or 3 sentences, not longer.
    User question: {question}
    """
   
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a personal assistant . Help users to learn  fundamentals and techniques. Recommend ideal  exercises to improve their  skills with detailed answers."
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  
            HumanMessagePromptTemplate.from_template(template),  
        ]
    )

    
    def get_conversation_chain():
        model = prepare_model()
        db = prepare_rag()
        llm = ChatOpenAI(model_name=model, temperature=0)
        retriever = db.as_retriever()
        
        memory = ConversationBufferWindowMemory(memory_key="chat_history",output_key="answer", return_messages=True, k=2)
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=retriever,
                            memory=memory,
                            condense_question_prompt=prompt,)
        
        #conversation_chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=template)
        
        print("Conversational Chain created for the LLM using the vector store")
        return conversation_chain

    conversation_chain=get_conversation_chain()
        
    def generate_text(user_input):
        start_time = time.time() 
        user_question = user_input
        response=conversation_chain({"question": user_question})
        response_time = time.time() - start_time
        print("Q: ",user_question)
        print("A: ",response['answer'])

        return response['answer']
    
    def clean_feedback():
        feedback.set_value("")
    
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
                    value="What is the best  product?",
                )
                generate_button = gr.Button("Ask  Assistant")
                

            with gr.Column(scale=2):
                response = gr.Textbox(
                    lines=4, label="Please wait few seconds for the answer...", interactive=False
                )
                
            # add textbox for feedback
            with gr.Column(scale=1):
                feedback = gr.Textbox(
                    lines=4,
                    label="Give Feedback / Fix the Answer",
                    interactive=True,
                    value="",
                )
                feedback_button = gr.Button("Give Feedback (one click)")

        gr.HTML("""
            <h2>Purpose of this session:</h2>
            <ul>
                <li>Please try to leave your feedbacks.</li>
                <li>If you find the answer good enough, please leave a feedback like "good answer" or similar. </li>
                <li>if you think the answer is bad or could be better please write the correct answer into the feedback section so we can improve the assistant.</li>
                <li>(examaple : why should i become better at ? -> assistant : [some answer]   possible feedback : "to get Lower Scores" is a better answer)</li>
            </ul>
        """)

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

