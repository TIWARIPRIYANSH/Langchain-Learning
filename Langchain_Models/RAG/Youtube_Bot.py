from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv

load_dotenv()
def fetch_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_obj = ytt_api.fetch(video_id, languages=["en"])
        # Flatten to plain text
        transcript = " ".join(chunk.text for chunk in transcript_obj)
        return transcript
    except Exception as e:
        print(f"Error fetching transcript: {e}")

video_id = "9QXCkMTbrSk"
transcript=fetch_transcript(video_id)

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
result=splitter.split_text(transcript)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store=Chroma.from_texts(
    texts=result,
    embedding=embeddings,
)

template=PromptTemplate(
    template="You are a useful assistant that tells if the video is about {topic_input}I am providing you video transcript{transcript}. Answer with yes or no.",
    input_variables=["topic_input", "transcript"]
)

retriever=vector_store.as_retriever(
    kwargs={"search_kwargs":{"k":3}}
)

model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash")

topic=input("Enter the topic you want to check in the video: ")
query=f"Is the video about {topic}?"
docs = retriever.invoke(query)
context = "\n\n".join(doc.page_content for doc in docs)
prompt=template.invoke({"topic_input":topic, "transcript":context})

response=model.invoke(prompt)
Parser=StrOutputParser()
final_output=Parser.parse(response.content)
print(final_output)






