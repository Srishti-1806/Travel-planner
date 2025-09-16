from langchain.chains import LLMChain, ConversationChain
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from fpdf import FPDF
import os
import speech_recognition as sr
import pyttsx3
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
urls = [
    "https://www.lonelyplanet.com/india",
    "https://www.tripadvisor.in/Attractions-g293860-Activities-India.html",
    "https://traveltriangle.com/blog/best-places-to-visit-in-india/",
    "https://www.holidify.com/country/india/places-to-visit.html"
]

loader = UnstructuredURLLoader(urls=urls)
raw_docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = splitter.split_documents(raw_docs)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory="./tour_chroma_db")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa_nlp = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

filter_prompt = PromptTemplate.from_template("""
Act as a professional tour planner. Based on the user's profile, plan the top 5 travel destinations in India or abroad. 
Include: destination name, highlights, best season, estimated budget, activities, nearby attractions, accommodation options,
and a match score (0–100) based on preferences.

USER PROFILE:
- Budget: {budget}
- Interests: {interests}
- Travel Duration: {duration}
- Travel Style: {style}
- Starting City: {city}

DESTINATION DATA:
{places}
""")

human_prompt = PromptTemplate.from_template("""
Create a warm and clear travel recommendation. For each suggested destination, include:
- Destination Name
- Why it matches the user
- Best Time to Visit
- Estimated Budget
- Top 3 Activities
- Accommodation Tip
- Match Score (0–100)

Finish with an inspiring note encouraging safe and fun travel.

DESTINATION DATA:
{filtered_places}
""")

memory = ConversationBufferWindowMemory(k=5)
filter_chain = LLMChain(prompt=filter_prompt, llm=llm)
response_chain = LLMChain(prompt=human_prompt, llm=llm)
convo = ConversationChain(llm=llm, memory=memory)

def save_pdf_report(title, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{title}\n\n{summary}")
    pdf.output("tour_plan.pdf")

def voice_input(prompt_text="Speak now..."):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(prompt_text)
        speak(prompt_text)
        audio = r.listen(source)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return input("Sorry, could not understand. Please type: ")
        except sr.RequestError:
            return input("Speech recognition error. Please type: ")

def generate_tour_plan(user_profile):
    query = f"Best destinations for budget {user_profile['budget']} with interests {user_profile['interests']}"
    retrieved_docs = retriever.get_relevant_documents(query)
    place_snippets = "\n".join([doc.page_content for doc in retrieved_docs])
    filter_input = {
        "budget": user_profile["budget"],
        "interests": ", ".join(user_profile["interests"]),
        "duration": user_profile["duration"],
        "style": user_profile["style"],
        "city": user_profile["city"],
        "places": place_snippets
    }
    filtered_places = filter_chain.run(filter_input)
    summarized = qa_nlp(filtered_places, max_length=800, min_length=300, do_sample=False)[0]['summary_text']
    final_summary = response_chain.run({"filtered_places": summarized})
    save_pdf_report("Your Tour Plan", final_summary)
    speak(final_summary)
    return final_summary

if __name__ == "__main__":
    print("🌍 Welcome to Yatra")
    speak("Welcome to Yatra")
    budget = voice_input("What is your budget for this trip (e.g., ₹50,000)? ")
    interests = voice_input("What are your main interests (e.g., beaches, trekking, heritage, shopping)? ").split(",")
    duration = voice_input("How many days do you plan to travel? ")
    style = voice_input("Preferred travel style (luxury, adventure, family, backpacking)? ")
    city = voice_input("From which city will you start your journey? ")
    user_profile = {
        "budget": budget.strip(),
        "interests": [i.strip() for i in interests],
        "duration": duration.strip(),
        "style": style.strip(),
        "city": city.strip()
    }
    response = generate_tour_plan(user_profile)
    print("\n===== YOUR TOUR PLAN =====\n")
    print(response)
    speak("A PDF summary has also been generated. Happy travels!")
    print("\nA PDF summary has also been generated and saved as 'tour_plan.pdf'.")
