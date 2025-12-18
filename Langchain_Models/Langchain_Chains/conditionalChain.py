from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv  
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Feed(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment")

ParserO = StrOutputParser()
Parser = PydanticOutputParser(pydantic_object=Feed)

template1 = PromptTemplate(
    template=(
        "Analyze the sentiment of this review and classify it as "
        "'positive' or 'negative':\n{review}\n{format_instructions}"
    ),
    input_variables=["review"],
    partial_variables={"format_instructions": Parser.get_format_instructions()}
)

outputP = PromptTemplate(
    template="Give Positive and constructive message based on positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

outputN = PromptTemplate(
    template="Give Negative and constructive message based on negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

# Correct: x is a Feed object → use x.sentiment
branching_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        outputP | model | ParserO,
    ),
    (
        lambda x: x.sentiment == "negative",
        outputN | model | ParserO,
    ),
    # Fallback
    RunnableLambda(lambda _: "Could not determine sentiment.")
)

chain = template1 | model | Parser | branching_chain

output = chain.invoke({
    "review": "The product quality is outstanding and exceeded my expectations"
})

print(output)
chain.get_graph().print_ascii()
