from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,load_prompt
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
     huggingfacehub_api_token=sec_key
   
)
model =ChatHuggingFace(llm=llm)
st.header("research tool")
paper_input =st.selectbox(
    "Select a research paper topic:",
    ["Artificial Intelligence", "Quantum Computing", "Climate Change", "Blockchain Technology"])
style_input=st.selectbox(
    "Select a writing style:",  
    ["Formal", "Informal", "Technical", "Narrative"])   
length_input=st.selectbox(
    "Select the length of the summary:",    
    ["Short", "Medium", "Long"])
prompt_template=load_prompt('template.json')
#prompt_template=prompt_template.format(paper_input=paper_input,style_input=style_input,length_input=length_input)
if st.button("Generate Summary"):
    chain = prompt_template | model
            
     # 6. Invoke with the dictionary of inputs
    output = chain.invoke({
         "paper_input": paper_input,
         "style_input": style_input,
         "length_input": length_input
            })
            
    st.write(output)