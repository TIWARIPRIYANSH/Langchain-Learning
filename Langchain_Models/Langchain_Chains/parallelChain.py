
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

parser=StrOutputParser()
template1=PromptTemplate(
    template="Provide a brief summary of the following text:\n{text}\n",
    input_variables=["text"]
)
template2=PromptTemplate(
    template="Create three quiz on the text \n {text}\n",
    input_variables=["text"]
)
template3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes with no markdown -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
   # model_kwargs={"temperature": 0.5, "max_length": 512}    
)
model1=ChatHuggingFace(llm=llm)
parallel_chain= RunnableParallel({
    'notes': template1 |model | parser,
    'quiz': template2 |model1 | parser,
    }
)
clean_markdown = RunnableLambda(
    lambda x: x.replace("**", "")
               .replace("##", "")
               .replace("* ", "")
               .replace("---", "")
)

final_chain= parallel_chain | template3 | model | parser | clean_markdown
output=final_chain.invoke({
    "text":"LangChain is an open-source framework designed to simplify the development of applications that leverage large language models (LLMs). It provides a modular architecture that allows developers to easily integrate various components such as prompt templates, memory management, and chains of operations. LangChain supports multiple LLM providers, enabling flexibility in choosing the underlying model. The framework is particularly useful for building complex applications like chatbots, question-answering systems, and content generation tools. By abstracting away much of the complexity involved in working with LLMs, LangChain empowers developers to focus on creating innovative solutions."
})
print(output)


