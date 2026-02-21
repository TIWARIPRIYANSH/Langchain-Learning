from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

doc=PyPDFLoader("C:\LANGCHAIN\Langchain_Models\LTI.pdf").load()
splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
result=splitter.split_documents(doc)
print(result[0])
