from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#it is used to put other chats in the conversatiion 
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []

with open('chathistory.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)


prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

print(prompt)