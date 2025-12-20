from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_core.prompts import PromptTemplate       
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv  
load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

template1= PromptTemplate(
    template="Write an article on the {topic}\n",
    input_variables=["topic"]
)

template2=PromptTemplate(
    template="Provide key takeaways from the article on {topic}\n",
    input_variables=["topic"] 
)
outout1_chain=RunnableSequence(template1,model,parser)

Branch_chain=RunnableBranch(
    (lambda x:len(x.split())<100, RunnableSequence(template2,model,parser)),
     RunnablePassthrough()
)
final_chain=RunnableSequence(outout1_chain,Branch_chain)
print(final_chain.invoke({"topic":"Artificial Intelligence"}))