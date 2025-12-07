from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
chat_model = ChatOpenAI(model="gpt-3.5-turbo",temperature=2)
output = chat_model.invoke("what is the capital of India")
print(output)
print(output.content)#only answer