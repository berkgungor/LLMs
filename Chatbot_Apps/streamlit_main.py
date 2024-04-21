import streamlit as st
from operator import itemgetter
from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
# from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from huggingface_hub import hf_hub_download
import csv
import time

def record_data(user_data, assistant_data, response_time, feedback=""):
    csv_file = 'chatbot_data_record_streamlit.csv'
    column_names = ['User', 'Assistant', 'ResponseTime','Feedback']

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


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


@st.cache_resource
def create_chain(system_prompt):
    # --- Disabled ---
    # A stream handler to direct streaming output on the chat screen.
    # This will need to be handled somewhat differently.
    # But it demonstrates what potential it carries.
    # stream_handler = StreamHandler(st.empty())

    # --- Disabled ---
    # Callback manager is a way to intercept streaming output from the
    # LLM and take some action on it. Here we are giving it our custom
    # stream handler to make it appear as if the LLM is typing the
    # responses in real time.
    # callback_manager = CallbackManager([stream_handler])

    (repo_id, model_file_name) = ("",
                                  "ggml-model18-q4.gguf")

    model_path = hf_hub_download(repo_id=repo_id,
                                 filename=model_file_name,
                                 repo_type="model")

    # n_gpu_layers, n_batch, and n_ctx are for GPU support.
    # When not set, CPU will be used.
    # set 1 for mac m2, and higher numbers based on your GPU support
    llm = LlamaCpp(
            model_path=model_path,
            temperature=0.6,
            max_tokens=200,
            top_p=1,
            # callback_manager=callback_manager,
            n_gpu_layers=20,
            n_batch=512,
            n_ctx=4096,
            verbose=False,
            streaming=True,
            stop=["Human:","###"]
            )
    print(llm)

    prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{human_input}"),
            ("ai", ""),
        ])

    # Conversation buffer memory
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)

    def save_memory(inputs_outputs):
        inputs = {"human": inputs_outputs["human"]}
        outputs = {"ai": inputs_outputs["ai"]}
        memory.save_context(inputs, outputs)

    def debug_memory():
        print("\n", "#"*10, "\n")
        print(memory.load_memory_variables({}))
        print("\n", "#"*10, "\n")

    def extract_response(chain_response):
        # debug_memory()
        return chain_response["ai"]

    llm_chain = {
            "human_input": RunnablePassthrough(),
            "chat_history": (
                RunnableLambda(memory.load_memory_variables) |
                itemgetter("chat_history")
            )
        } | prompt | llm
    chain_with_memory = RunnablePassthrough() | {
                "human": RunnablePassthrough(),
                "ai": llm_chain
            } | {
                "save_memory": RunnableLambda(save_memory),
                "ai": itemgetter("ai")
            } | RunnableLambda(extract_response)

    return chain_with_memory


st.set_page_config(
    page_title=" Golf Assistant!"
)

# Create a header element
st.header(" Golf Assistant! (Please wait 4-5 seconds after sent the message.)")

prompt = """You are a personal assistant . Help users to learn  fundamentals and techniques. Recommend ideal  exercises to help users to improve their  skills."""

llm_chain = create_chain(prompt)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I am your  assistant. How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_prompt := st.chat_input("Your message here", key="user_input"):


    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    # Add input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)

    start_time = time.time()
    response = llm_chain.invoke(user_prompt)
    end_time = time.time()
    response_time = end_time - start_time
    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    # Add the response to the chat window
    with st.chat_message("assistant"):
        st.markdown(response)

    record_data(user_prompt, response, response_time, "")