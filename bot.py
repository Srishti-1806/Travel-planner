import streamlit as st
from langchain.chains import LLMChain, RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from fpdf import FPDF
import os
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pyttsx3
import speech_recognition as sr

load_dotenv()

engine = pyttsx3.init()

def speak_text(text: str):
    """Text to speech"""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"TTS Error: {e}")

def listen_voice() -> str:
    """Voice input with fallback"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak now")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I could not understand."
        except sr.RequestError:
            return "Speech recognition service error."
        except Exception as e:
            return f"Error: {e}"

@st.cache_resource
def load_data():
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
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory="./tour_chroma_db"
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    qa_nlp = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    return qa_chain, qa_nlp, retriever, llm

qa_chain, qa_nlp, retriever, llm = load_data()

def save_pdf_report(title: str, summary: str, filename="tour_plan.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{title}\n\n{summary}")
    pdf.output(filename)
    return filename

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

def generate_tour_plan(user_profile: dict) -> str:
    try:
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

        summarized = qa_nlp(
            filtered_places,
            max_length=800,
            min_length=300,
            do_sample=False
        )[0]['summary_text']

        final_summary = response_chain.run({"filtered_places": summarized})

        pdf_file = save_pdf_report("Your Tour Plan", final_summary)
        return final_summary, pdf_file
    except Exception as e:
        return f"⚠️ Error generating plan: {e}", None

st.set_page_config(page_title="🌍 Yatra Bot", page_icon="🧳", layout="wide")

st.title("🌍 Yatra – AI Travel Planner")
st.write("Plan your perfect trip with AI recommendations ✈️🏞️")

with st.form("user_profile_form"):
    budget = st.text_input("💰 Budget (e.g., ₹50,000)")
    interests = st.text_input("🎯 Interests (comma-separated, e.g., beaches, trekking, shopping)")
    duration = st.text_input("🗓️ Travel Duration (e.g., 7 days)")
    style = st.selectbox("🎒 Travel Style", ["Luxury", "Adventure", "Family", "Backpacking"])
    city = st.text_input("🏙️ Starting City")

    submitted = st.form_submit_button("Generate Tour Plan")

if submitted:
    user_profile = {
        "budget": budget.strip(),
        "interests": [i.strip() for i in interests.split(",")],
        "duration": duration.strip(),
        "style": style.strip(),
        "city": city.strip()
    }
    st.session_state["profile"] = user_profile

    with st.spinner("✨ Creating your tour plan..."):
        response, pdf_file = generate_tour_plan(user_profile)

    st.subheader("✅ Your Personalized Tour Plan")
    st.write(response)

    if pdf_file:
        with open(pdf_file, "rb") as f:
            st.download_button("📥 Download PDF Plan", f, file_name="tour_plan.pdf")

    speak_text(response)


if "profile" in st.session_state:
    st.subheader("💬 Chat with Yatra Bot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    col1, col2 = st.columns([4,1])
    with col1:
        user_query = st.chat_input("Type your query or ask about destinations...")
    with col2:
        if st.button("🎙️ Speak"):
            user_query = listen_voice()
            st.chat_message("user").markdown(f"🎤 {user_query}")

    if user_query:
        if user_query.lower() in ["exit", "quit", "bye"]:
            st.chat_message("assistant").markdown("👋 Goodbye! Have a safe journey!")
        else:
            st.session_state["messages"].append({"role": "user", "content": user_query})

            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = qa_chain.run(user_query)
                    st.markdown(answer)
                    speak_text(answer)

            st.session_state["messages"].append({"role": "assistant", "content": answer})
