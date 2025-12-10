from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)
Model=ChatHuggingFace(llm=llm)


template2=PromptTemplate(
   template="You are a helpful assistant that summarizes research papers on {paper_input} in a {style_input} style. Provide a {length_input} summary highlighting key points and findings.",
   input_variables=["paper_input","style_input","length_input"]
)

prompt=template2.invoke({"paper_input":"Artifical_Intelligence",
                        "style_input":"Formal", 
                        "length_input":"Short"})

result=Model.invoke(prompt)
print(result.content)

 