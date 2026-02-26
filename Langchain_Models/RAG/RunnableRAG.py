from langchain_community.document_loaders import YoutubeLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough,
)
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# 1️⃣ Load & Split Transcript - Ingestion - INDEX
# ---------------------------
input_url = input("Enter the YouTube video URL: ")
loader = YoutubeLoader.from_youtube_url(input_url, language="en")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splitted_docs = splitter.split_documents(docs)

# ---------------------------
# 2️⃣ Create Vector Store
# ---------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=splitted_docs,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# ---------------------------
# 3️⃣ Helper: Join Docs
# ---------------------------
def join_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------------------
# 4️⃣ RunnableParallel
# ---------------------------
parallel_chain = RunnableParallel(
    context = retriever | RunnableLambda(join_docs),#we have to make join_doc as a runnable to use it in parallel chain as retriever will return list of docs and we have to convert it into string before passing to the prompt
    topic_input = RunnablePassthrough()
)

# ---------------------------
# 5️⃣ Prompt
# ---------------------------
template = PromptTemplate(
    template="""
You are an expert content classifier.

Based ONLY on the provided transcript context, determine whether the video is about {topic_input}.

Transcript Context:
---------------------
{context}
---------------------

If the topic is clearly discussed, answer YES.
If not discussed, answer NO.
Answer strictly with YES or NO.
""",
    input_variables=["topic_input", "context"]
)

# ---------------------------
# 6️⃣ Model + Parser (RunnableSequence)
# ---------------------------
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

sequence_chain = template |model | parser

# ---------------------------
# 7️⃣ Final Combined Chain
# ---------------------------
final_chain = parallel_chain | sequence_chain

topic = input("Enter the topic you want to check in the video: ")

response = final_chain.invoke(topic)

print("\nFinal Answer:", response)