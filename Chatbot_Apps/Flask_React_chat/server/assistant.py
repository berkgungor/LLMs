import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import re
#import STT 
#import TTS
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#3.5 normal : ftjob-hjLV1XIM6ccPjzx6zTpAYw5z
#3.5 class : ft:gpt-3.5-turbo-1106:personal:class2:99ElJsUA

class Assistant:
    def __init__(self, modelType):
        self.chat_history = []
        self.OPENAI_API_KEY = ""
        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        self.job_id = 'ftjob-Wg2imnR2U6bjgfqlYky4UXhN'
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.finetuned_job = self.client.fine_tuning.jobs.retrieve(self.job_id)
        if modelType == "pv-assistant":
            modelType = self.finetuned_job.fine_tuned_model
        
        #self.finetuned_model = self.finetuned_job.fine_tuned_model
        self.llm_model = ChatOpenAI(model_name=modelType, temperature=0.7, max_tokens=200)
        print(self.llm_model)
        self.TTS = False

    def reload_chat(self):
        print("chat reloaded")
        self.chat_history = []
        
    
    def get_document(self,pdf_file):
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(docs, embedding = embeddings)
        return vector_store

    def assistant_type(self,assistantType):
        if assistantType == "PV":
            prompt = ChatPromptTemplate.from_messages([
            ("system", """Your name is  Assistant. You are a personal assistant .You help users to learn  fundamentals and techniques with detailed answers. 
              helps people to practice  with the help of a projector and a indoor green platform so you might be asked to project a  practice on green.
             if you do not know the answer, do not make up an answer, say that you do not know the answer.
             Answer within 3 sentences to the user's question based on the context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")
            ])
            
        elif assistantType == "marketing":
            prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a marketing assistant. You help users to develop their marketing strategies. The main products that you try to do marketing about is  simulator called .
             if you do not know the answer, do not make up an answer, say that you do not know the answer.
             Answer within 3 sentences to the user's question based on the context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")
            ])
        
        elif assistantType == "sales":
            prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a sales assistant. You help other sales agents to develop their sales strategies. The main products that you try to sell is  simulator called .
             if you do not know the answer, do not make up an answer, say that you do not know the answer.
             Answer within 3 sentences to the user's question based on the context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")
            ])
        
        elif assistantType == "software":
            prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a software assistant.You help users to write codes and fix their bugs.
             if you do not know the answer, do not make up an answer, say that you do not know the answer.
             Answer within 3 sentences to the user's question based on the context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
            ("system", """Your name is  Assistant. You are a personal assistant .You help users to learn  fundamentals and techniques with detailed answers. 
              helps people to practice  with the help of a projector and a indoor green platform so you might be asked to project a  practice on green.
             if you do not know the answer, do not make up an answer, say that you do not know the answer.
             Answer within 3 sentences to the user's question based on the context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")
            ])
        return prompt
        
    def history_and_retriever(self, vector_store, prompt):  
        
        chain = create_stuff_documents_chain(llm = self.llm_model, prompt = prompt)

        #similarity = vector_store.similarity_search()
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        
        history_aware_retriever = create_history_aware_retriever(llm=self.llm_model, retriever = retriever, prompt= retriever_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, chain)
        
        return retrieval_chain
    
    
    def normal_llm(self, assistantType):
        
        if assistantType == "marketing":
            system_prompt = """You are a marketing assistant. You help users to develop their marketing strategies. The main products that you try to do marketing about is  simulator called .
                if you do not know the answer, do not make up an answer, say that you do not know the answer. Question: {input}
                Answer:"""
        elif assistantType == "sales":
            system_prompt = """You are a sales assistant. You help other sales agents to develop their sales strategies. The main products that you try to sell is  simulator called .
                if you do not know the answer, do not make up an answer, say that you do not know the answer. Question: {input}
                Answer:"""
        elif assistantType == "software":
            system_prompt = """You are a software assistant. You help users to write codes and fix their bugs.
                if you do not know the answer, do not make up an answer, say that you do not know the answer. Question: {input}
                Answer:"""

        normal_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        llm_chain = LLMChain(prompt=normal_prompt, llm=self.llm_model)

        return llm_chain
    

    def prepare_conversation(self, main_chain, question):
        response = main_chain.invoke({
            "input" : question,
            "chat_history" : self.chat_history
        })
        print(response)
        return response

    def extract_exercise_type(self, answer):
        pattern_after = r'<<(.*)'
        pattern_before = r'^(.*?)<'
        main_text = re.search(pattern_before, answer)
        exercise = re.search(pattern_after, answer)
        answer = main_text.group(1)
        return answer, exercise


    def call_assistant(self, user_input, file_name, assistantType):
        print("Assistant type: ", assistantType)
        print("User input: ", user_input)
        main_prompt = self.assistant_type(assistantType)
        if file_name!= "":
            vector_store = self.get_document(file_name)
            history_aware_chain = self.history_and_retriever(vector_store,main_prompt)
            response = self.prepare_conversation(history_aware_chain, user_input)
            response = response["answer"]
        else:
            print("Normal LLM")
            normal_chain = self.normal_llm(assistantType)
            response = self.prepare_conversation(normal_chain, user_input)
            response = response["text"]
            
        if self.TTS == True:
            voice_input = response
            self.speech_output(voice_input)
        #answer, exercise = self.extract_exercise_type(response)
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response))
        if len(self.chat_history) > 4:
            self.chat_history = self.chat_history[-4:]
        return response
            
"""    def speech_output(self, voice_input):
        tts = TTS.SpeakOut()
        tts.labs(voice_input)
        
    def speech_input(self):
        transcriber  = STT.FasterSpeech()
        transcription = transcriber.run_transcriber()
        if self.TTS == True:
            self.speech_output(transcription)
        return transcription"""
    