
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv  
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",     
    task="text-generation",
)
model=ChatHuggingFace(llm=llm)
messages=[
    SystemMessage(content="You are a helpful assistant that write love poems in 50 words."),
    HumanMessage(content="Write a Heartbreaking love poem about a lost love.")
]

messages.append(AIMessage(content=model.invoke(messages).content))
print(messages)