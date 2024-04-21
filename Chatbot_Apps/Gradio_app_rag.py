from langchain_core.prompts import format_document  
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from langchain.memory import ConversationBufferMemory
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
    pdf_docs=["_RAG_2.pdf"]

    (repo_id, model_file_name) = ("",
                                    "ggml-model27-q4.gguf")

    model_path = hf_hub_download(repo_id=repo_id,
                                    filename=model_file_name,
                                    repo_type="model")

    template = """
    You are a personal assistant  and you also enlight users about  products. 
    Help users to learn  fundamentals and techniques with detailed answers. Recommend ideal  exercises to help users to improve their  skills.
    ### Instruction:{question}
    ### Response:
    """

    def prepare_docs(pdf_docs):
        docs = []
        metadata = []
        content = []

        for pdf in pdf_docs:

            pdf_reader = PyPDF2.PdfReader(pdf)
            for index, text in enumerate(pdf_reader.pages):
                doc_page = {'title': pdf + " page " + str(index + 1),
                            'content': pdf_reader.pages[index].extract_text()}
                docs.append(doc_page)
        for doc in docs:
            content.append(doc["content"])
            metadata.append({
                "title": doc["title"]
            })
        print("Content and metadata are extracted from the documents")
        return content, metadata

    def get_text_chunks(content, metadata):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512,
            chunk_overlap=40,
        )
        split_docs = text_splitter.create_documents(content, metadatas=metadata)
        print(f"Documents are split into {len(split_docs)} passages")
        return split_docs

    def ingest_into_vectordb(split_docs):
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
        db = FAISS.from_documents(split_docs, embeddings)

        DB_FAISS_PATH = 'vectorstore/_FAISS'
        db.save_local(DB_FAISS_PATH)
        return db


    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    def get_conversation_chain(vectordb):
        llamacpp = LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        n_gpu_layers=512,
        n_batch=30,
        max_tokens=90,
        top_p=1,
        stop=["Human:","HUMAN", "###","You are a personal assistant","Hi!"],
        callback_manager=callback_manager,
        n_ctx=8000)

        retriever = vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 2})
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True, output_key='answer')
        print(">>> memory: ", memory)
        conversation_chain = (ConversationalRetrievalChain.from_llm
                            (llm=llamacpp,
                            retriever=retriever,
                            #condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                            memory=memory,
                            return_source_documents=True))
        print("Conversational Chain created for the LLM using the vector store")
        return conversation_chain


    content, metadata = prepare_docs(pdf_docs)
    split_docs = get_text_chunks(content, metadata)
    vectordb=ingest_into_vectordb(split_docs)
    conversation_chain=get_conversation_chain(vectordb)


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
            Golf Assistant v1.0
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

