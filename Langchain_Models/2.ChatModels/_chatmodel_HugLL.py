import os
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm=HuggingFaceEndpoint(
     repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=sec_key
)
model=ChatHuggingFace(llm=llm)
result= model.invoke("What is the capital of UP?")
print(result.content)