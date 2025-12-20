from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv  
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

def word_length(x):
    return len(x.split())

template= PromptTemplate(
    template=(
        "Print a joke on the topic:{topic}"),
        input_variables=["topic"]
    
)

joke_chain=RunnableSequence(template,model,parser)

parallelchain=RunnableParallel({
     'joke': RunnablePassthrough(),#it just passes the input as it is
     'word_count':RunnableLambda(word_length) #It converts the python function to runnable
}
)
final_chain=RunnableSequence(joke_chain,parallelchain)
print(final_chain.invoke({"topic":"computers"}))

