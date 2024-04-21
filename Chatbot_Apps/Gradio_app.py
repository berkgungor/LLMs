from operator import itemgetter
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
# from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from huggingface_hub import hf_hub_download
import gradio as gr
import csv
import time 


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


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



def main():
    def create_chain(system_prompt):
        """(repo_id, model_file_name) = ("",
                                    "ggml-model18-q4.gguf")

        model_path = hf_hub_download(repo_id=repo_id,
                                    filename=model_file_name,
                                    repo_type="model")"""

        # model_path = ""

        # n_gpu_layers, n_batch, and n_ctx are for GPU support.
        # When not set, CPU will be used.
        # set 1 for mac m2, and higher numbers based on your GPU support
        llm = LlamaCpp(
            model_path="/home//Repository/AI_Coach/llama.cpp/models/7B/merged_model_27/ggml-model27-q4.gguf",
            temperature=0.7,
            max_tokens=80,
            top_p=1,
            # callback_manager=callback_manager,
            n_gpu_layers=20,
            n_batch=512,
            n_ctx=32096,
            verbose=False,
            streaming=True,
            stop=["Human:","HUMAN", "###","You are a personal assistant","Hi!"]
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

    prompt = """You are a personal assistant . Help users to learn  fundamentals and techniques. Recommend ideal  exercises to help users to improve their  skills."""
    llm_chain = create_chain(prompt)
    
    def generate_text(prompt):
        start_time = time.time()
        output = llm_chain.invoke(prompt)
        response_time = time.time() - start_time
        feedback = ""
        record_data(prompt, output, feedback, response_time)
        return output

    
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
                    value="What is Ladder Drill?",
                )
                generate_button = gr.Button("Ask  Assistant")
                
                
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