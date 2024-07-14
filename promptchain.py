from langchain_huggingface import HuggingFaceEndpoint
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.sequential import SequentialChain,SimpleSequentialChain
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from langchain.memory import ConversationBufferMemory
from constants import huggingface_api

import streamlit as st
#streamlit framework
st.title("Famous People all Around the World")
input_text=st.text_input("Search the person u want")
 #Accessing the LLM
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api
repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=huggingface_api)
#Prompt Templates and BufferMemory
profession_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory=ConversationBufferMemory(input_key='profession',memory_key='chat_history')
desc_memory=ConversationBufferMemory(input_key='dob',memory_key='description_history') 

input_prompt=PromptTemplate(input_variables=['name'],template="Tell me about {name}")
chain=LLMChain(llm=llm,prompt=input_prompt,verbose=True,output_key='profession',memory=profession_memory)

second_prompt=PromptTemplate(input_variables=['profession'],template="What is his {profession}?")
chain2=LLMChain(llm=llm,prompt=second_prompt,verbose=True,output_key='dob',memory=dob_memory)

third_prompt=PromptTemplate(input_variables=['dob'],template="Mention 5 things that happened on {dob}")
chain3=LLMChain(llm=llm,prompt=third_prompt,verbose=True,output_key='description',memory=desc_memory)
                            
parent_chain=SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['profession','dob','description'],verbose=True)

#Displaying the Results
if input_text:
    st.write(parent_chain({'name':input_text}))
    with st.expander('Profession'):
        st.info(profession_memory.buffer)
    with st.expander('Major Events'):
        st.info(desc_memory.buffer)
