import os
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from huggingface_hub import hf_hub_download


def call_model():
    (repo_id, model_file_name) = ("",
                                    "ggml-model18-q4.gguf")

    model_path = hf_hub_download(repo_id=repo_id,
                                filename=model_file_name,
                                repo_type="model")

    # model_path = ""

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
        stop=["Human:", "###"]
    )
    
    return llm
    

loader = PyPDFLoader("C:\Users\berkg\Repository\-ai-coach\FineTune\_data.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(
    search_type="similarity", search_kwargs={"k": 2}
)
# create a chain to answer questions
qa = RetrievalQA.from_chain_type(
    llm=call_model(),
    chain_type="map_reduce",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
)

print(qa)
