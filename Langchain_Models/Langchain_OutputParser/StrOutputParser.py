from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv  
load_dotenv()
model = ChatGoogleGenerativeAI(
model="gemini-2.5-flash"
)
Parser=StrOutputParser()
template1=PromptTemplate(
    template="Write a detailed text funny story about{topic}",
    input_variables=["topic"]
)
template1_prompt=template1.invoke({"topic":"Virat Kohli"})
output1=model.invoke(template1_prompt)
print(output1.content)
template2=PromptTemplate(
    template="Summarize the following text in 3 lines:{text}",
    input_variables=["text"]
)
template2_prompt=template2.invoke({"text":output1})
output2=model.invoke(template2_prompt)
Parser=Parser.parse(output2.content)
print(Parser)

# output= template1 | model | template2 | model | Parser
# print(output.invoke({"topic":"Virat Kohli"}))


