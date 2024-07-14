from langchain_huggingface import HuggingFaceEndpoint
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from constants import huggingface_api
import streamlit as st
#streamlit framework
st.title("Langchain Demo")
input_text=st.text_input("Search the topic u want")
 #Accessing the LLM
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api
repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=huggingface_api)

if input_text:
    st.write(llm(input_text))
