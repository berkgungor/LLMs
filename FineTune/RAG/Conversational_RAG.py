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
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import nest_asyncio
from operator import itemgetter
#memory libraries
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate



model_name='mistralai/Mistral-7B-Instruct-v0.2'
#model_name= "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

use_4bit = True
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
use_nested_quant = False #(double quantization)

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)


mistral_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)

standalone_query_generation_pipeline = pipeline(
    model=mistral_model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.0,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=200,
)
standalone_query_generation_llm = HuggingFacePipeline(pipeline=standalone_query_generation_pipeline)

response_generation_pipeline = pipeline(
    model=mistral_model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.7,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=200,
)
response_generation_llm = HuggingFacePipeline(pipeline=response_generation_pipeline)

# Load all the text
nest_asyncio.apply()
# Articles to index
articles = ["https://www..com/blog/the-need-for-speed/",
            "https://www..com/blog/improve-your-green-reading-with-these-easy-steps/",
            "https://www..com/blog/make-the-most-out-of-your--practice/",
            "https://www..com/products/indoor/features/",
            "https://www..com/blog/strokes-gained-/",
            "https://www..com/blog/get-in-the-hole/",
            "https://www..com/blog/visual-cues/",
            "https://www..com/blog/your-trackman-for-/",
            "https://www..com/blog/putter-fitting-with-martin-stecher/"]

loader = AsyncChromiumLoader(articles)
docs = loader.load()
# Converts HTML to plain text 
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
# Chunk text
text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=0) # @TODO: Make this configurable and minimize the chunk size
chunked_documents = text_splitter.split_documents(docs_transformed)
# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
retriever = db.as_retriever(k = 1)

# CONVERSATIONAL CHAT
_template = """
[INST] 
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language, that can be used to query a FAISS index. This query will be used to retrieve documents with additional context. 

Let me share a couple examples that will be important. 
If you do not see any chat history, you MUST return the "Follow Up Input" as is:
```
Chat History:

Follow Up Input: what are fundamentals of golf?
Standalone Question:
what are fundamentals of golf?
```
If this is the second question onwards, you should properly rephrase the question like this:

```
Chat History:
what are fundamentals of golf?
AI: 
Fundamentals of golf are speed, line, and green reading.
Follow Up Input: Can you explain them?
Standalone Question:
Can you explain the fundamentals of golf?
```
Now, with those examples, here is the actual chat history and input question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:
[your response here]
[/INST] 
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """
[INST] 
You are an AI golf assistant that helps users to improve their  skills. Answer the question based only on the following context:
{context}

Question: {question}
[/INST] 
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# Instantiate ConversationBufferMemory
memory = ConversationBufferMemory(
 return_messages=True, output_key="answer", input_key="question"
)

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | standalone_query_generation_llm,
}
# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | response_generation_llm,
    "question": itemgetter("question"),
    "context": final_inputs["context"]
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer

import time
def call_conversational_rag(question, chain, memory):
    """
    Calls a conversational RAG (Retrieval-Augmented Generation) model to generate an answer to a given question.

    This function sends a question to the RAG model, retrieves the answer, and stores the question-answer pair in memory 
    for context in future interactions.

    Parameters:
    question (str): The question to be answered by the RAG model.
    chain (LangChain object): An instance of LangChain which encapsulates the RAG model and its functionality.
    memory (Memory object): An object used for storing the context of the conversation.

    Returns:
    dict: A dictionary containing the generated answer from the RAG model.
    """
    start = time.time()
    
    # Prepare the input for the RAG model
    inputs = {"question": question}

    # Invoke the RAG model to get an answer
    result = chain.invoke(inputs)
    
    # Save the current question and its answer to memory for future context
    memory.save_context(inputs, {"answer": result["answer"]})
    end = time.time()
    print("response time :",end - start)
    # Return the result
    return result

while True:
    print("-----------------------------------------------------")
    question = input("User: ")
    text = call_conversational_rag(question, final_chain, memory)
    print(text['answer'])