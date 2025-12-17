from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

load_dotenv()

class StorySummary(BaseModel):
    title:str=Field(description="The title of the story")
    summary:str=Field(description="A brief summary of the story in 5 minutes")
    Humor_Level:int=Field(gt=4 ,lt=10, description="The level of humor in the story")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    #repo_id="google/gemma-2-2b-it",
    task="text-generation",
)
Model = ChatHuggingFace(llm=llm)

Pydantic_Parser = PydanticOutputParser(pydantic_object=StorySummary)
prompt_template = PromptTemplate(
    template="You are a creative writer. Write a funny story about {topic} with a catchy title. Ensure the humor level is high.\n {format_instructions}", 
    input_variables=["topic"],
    partial_variables={"format_instructions":Pydantic_Parser.get_format_instructions()}
) 
# output_prompt = prompt_template.invoke({"topic":"an adventurous cat exploring the city"})
# output = Model.invoke(output_prompt)
# output_parsed = Pydantic_Parser.parse(output.content)
output=prompt_template | Model | Pydantic_Parser

out=output.invoke({"topic":"an adventurous cat exploring the city"})
print(out.title)
