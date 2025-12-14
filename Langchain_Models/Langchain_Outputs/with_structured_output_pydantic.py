from pydantic import BaseModel,Field
from typing import Optional,Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from torch import negative, positive
load_dotenv()

class Document(BaseModel):
    Key_themes:list[str]=Field(description="Write down all the key themes present in the document.")
    summary:Optional[str]=Field(default=None,description="Provide a concise summary of the document."   )
    sentiment:Literal['positive','negative']=Field(description="Overall sentiment of the document ('positive' or 'negative').")
    pros:Optional[list[str]]=Field(default=None,description="List the positive aspects mentioned in the document.")
    cons:Optional[list[str]]=Field(default=None,description="List the negative aspects mentioned in the document.")

model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
model_output=model.with_structured_output(Document)
output=model_output.invoke( 
    """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver."""
)
print(output)
print(output.cons)