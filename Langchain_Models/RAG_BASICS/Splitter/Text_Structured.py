from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("C:\LANGCHAIN\Langchain_Models\LTI.pdf")
doc=loader.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
result=splitter.split_documents(doc)
print(result[0].page_content)