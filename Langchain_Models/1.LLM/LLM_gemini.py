from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatVertexAI(
    model="gemini-1.5-flash"   )

result = llm.invoke("What is the capital of UP?")
print(result)
