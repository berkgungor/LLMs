
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler
)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

main_template = """You are a personal assistant  and you also enlight users about  products. Help users to learn  fundamentals and techniques. Recommend ideal  exercises to help users to improve their  skills.
        ### Instruction: {question}
        ### Response:
        """

prompt_main = PromptTemplate(template=main_template, input_variables=["question"])
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

loader = PyPDFLoader("/home//Repository/AI_Coach/RAG/_RAG.pdf")
documents = loader.load()

(repo_id, model_file_name) = ("",
                                  "ggml-model18-q4.gguf")

model_path = hf_hub_download(repo_id=repo_id,
                                 filename=model_file_name,
                                 repo_type="model")
llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=512,
        n_batch=30,
        n_ctx=6000,
        max_tokens=200,
        temperature=0,
        # callback_manager=callback_manager,
        verbose=True,
        streaming=True,
        )

llm_chain_main = LLMChain(prompt=prompt_main, llm=llm)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
hf_embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

db = FAISS.from_documents(docs, hf_embedding)
db.save_local("_FAISS")
# load from local
db = FAISS.load_local("_FAISS/", embeddings=hf_embedding)

template_1 = '''Context: {context}
Based on the Context, provide an answer as a  coach for following question
### Instruction:{question}
'''

question = input("User: ")
search = db.similarity_search(question, k=2)
prompt = PromptTemplate(input_variables=["context", "question"], template= template_1)
final_prompt = prompt.format(question=question, context=search)
rag_out = llm_chain_main.run(final_prompt)
print(rag_out)