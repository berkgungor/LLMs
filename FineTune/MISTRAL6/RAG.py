import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from datasets import load_dataset
from peft import LoraConfig, PeftModel

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
import nest_asyncio
nest_asyncio.apply()

# Articles to index
articles = ["https://www..com/blog/the-need-for-speed/",
            "https://www..com/blog/improve-your-green-reading-with-these-easy-steps/",
            "https://www..com/blog/make-the-most-out-of-your--practice/",
            "https://www..com/products/indoor/features/"]

# Scrapes the blogs above
loader = AsyncChromiumLoader(articles)
docs = loader.load()

model_name='mistralai/Mistral-7B-Instruct-v0.1'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.4,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=200,
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Converts HTML to plain text 
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

# Chunk text
text_splitter = CharacterTextSplitter(chunk_size=100, 
                                      chunk_overlap=0)
chunked_documents = text_splitter.split_documents(docs_transformed)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents, 
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

retriever = db.as_retriever()

prompt_template = """
### [INST] Instruction: You are a personal assistant named  coach and your area of expertise is golf. Answer the questions. Here is context to help:

{context}

### QUESTION:
{question} [/INST]
 """

# Create prompt from prompt template 
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

def llm_chain(user_input):
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)
    llm_chain.invoke({"context": "", "question": user_input})
    #print("=================== LLM CHAIN ==================================\n")
    #print(result_llm['text'])

    rag_chain = ( 
    {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )
    result = rag_chain.invoke(user_input)
    print("=================== RAG CHAIN ==================================\n")
    print(result['text'])

while True:
    user_input = input("User: ")
    llm_chain(user_input)