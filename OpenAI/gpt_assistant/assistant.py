import logging
import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
import time
import shelve

os.environ["OPENAI_API_KEY"] = ""
job_id = 'ftjob-G86G2vEwvVrDoq6CcaOcxKes'
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))



def upload_file(file_path):
    file = client.files.create(file=open(file_path, "rb"), purpose="assistants")
    return file

def create_assistant(file):
    
    assistant = client.beta.assistants.create(
        name=" Assistant",
        instructions="Your name is  Assistant. You are a personal assistant .\
            You help users to learn  fundamentals and techniques with detailed answers.",
        tools=[{"type": "retrieval"}],
        model="gpt-3.5-turbo",
        file_ids=[file.id],
    )
    
    return assistant


file = upload_file("_RAG_3.pdf")
#assistant = create_assistant(file)


def check_if_thread_exists(user_id):
    with shelve.open("threads.db") as threads_shelf:
        return threads_shelf.get(user_id, None)

def store_thread(user_id, thread_id):
    with shelve.open("threads.db", writeback=True) as threads_shelf:
        threads_shelf[user_id] = thread_id


def generate_response(message_body,user_id, name):
    
    thread_id = check_if_thread_exists(user_id)
    if thread_id is None:
        logging.info("Creating a new thread for {name}")
        thread = client.beta.threads.create()
        store_thread(user_id, thread.id)
        thread_id = thread.id
    else:
        logging.info(f"Using existing thread for {name}")
        thread = client.beta.threads.retrieve(thread_id)
        
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message_body,
    )
    
    new_message = run_assistant(thread)
    return new_message



def run_assistant(thread):
    assistant = client.beta.assistants.retrieve("asst_pV0uFTqBgkseCzRnKcoXEwqP")
    
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    
    while run.status != "completed":
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    new_message = messages.data[0].content[0].text.value
    logging.info(f"generated message: {new_message}")
    return new_message


new_message = generate_response("is trackman better than ?","123","berk")
print(new_message)
    