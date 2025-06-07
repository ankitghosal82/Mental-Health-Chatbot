import streamlit as st
import os
from dotenv import load_dotenv
import requests
import datetime
import speech_recognition as sr
import pyttsx3
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# Load environment variables (for API key)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Download NLTK data
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Streamlit UI setup
st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("🧠 Mental Health Chatbot")
st.markdown("Talk about your feelings. I'm here to listen and support you anonymously. ❤️")

# Mood check-in input
st.markdown("### ☀ Daily Mood Check-in")
user_mood = st.text_input("How are you feeling today?")

def save_mood_log(mood, sentiment_score):
    log_file = "mood_log.csv"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[timestamp, mood, sentiment_score]], columns=["timestamp", "mood", "sentiment"])
    if os.path.exists(log_file):
        entry.to_csv(log_file, mode='a', header=False, index=False)
    else:
        entry.to_csv(log_file, index=False)

# Save mood if user enters it
if user_mood:
    sentiment = sia.polarity_scores(user_mood)
    score = sentiment['compound']
    save_mood_log(user_mood, score)

    if score >= 0.05:
        st.success("You seem positive today! 😊")
    elif score <= -0.05:
        st.error("You sound a bit down. I'm here for you. ❤")
    else:
        st.info("You're feeling neutral today. Let's talk more!")

# Voice input
st.markdown("### 🎙 Voice Input (Optional)")
if st.button("🎧 Speak Now"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source)
    try:
        user_input = r.recognize_google(audio)
        st.write("You said:", user_input)
    except sr.UnknownValueError:
        st.warning("Sorry, I didn't understand that.")
        user_input = ""
else:
    user_input = st.text_input("💬 Type your message:")

# Function to call OpenRouter API for AI response
def chat_with_openrouter(user_input):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    system_prompt = (
        "You are a compassionate, non-judgmental anonymous mental health assistant. "
        "Whenever the user shares something, provide supportive and personalized advice. "
        "Speak kindly and encourage them with motivational words and mental wellness tips. "
        "Be gentle, warm, and helpful, like a friend who deeply understands."
    )
    payload = {
        "model": "mistralai/mixtral-8x7b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# Generate bot response and speak it
if user_input:
    with st.spinner("Thinking..."):
        try:
            response = chat_with_openrouter(user_input)
            st.markdown("🤖 Bot: " + response)
            speak(response)
        except Exception as e:
            st.error(f"Failed to connect to AI. Check your API key or internet.\n{e}")

# Show mood over time chart
if os.path.exists("mood_log.csv"):
    st.markdown("### 📈 Mood Over Time")
    df = pd.read_csv("mood_log.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")
    st.line_chart(df.set_index("timestamp")["sentiment"])
