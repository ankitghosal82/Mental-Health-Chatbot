

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
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv() 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Text-to-speech engine
tts_engine = pyttsx3.init() 
tts_engine.setProperty('rate', 150)

def speak(text): 
    tts_engine.say(text) 
    tts_engine.runAndWait()

# Download sentiment lexicon
nltk.download('vader_lexicon') 
sia = SentimentIntensityAnalyzer()

# Streamlit app setup
st.set_page_config(page_title="Anonymous Mental Health Chatbot", layout="centered") 
st.title("🧠 Anonymous Mental Health Chatbot") 
st.markdown("Talk about your feelings. I'm here to listen anonymously. ❤️")

# Mood check-in section
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

# OpenRouter integration
def chat_with_openrouter(user_input): 
    url = "https://openrouter.ai/api/v1/chat/completions" 
    headers = { 
        "Authorization": f"Bearer {OPENROUTER_API_KEY}", 
        "Content-Type": "application/json" 
    } 
    payload = { 
        "model": "mistralai/mixtral-8x7b", 
        "messages": [ {"role": "user", "content": user_input} ] 
    } 
    response = requests.post(url, headers=headers, json=payload) 
    return response.json()['choices'][0]['message']['content']

# Respond to user input
if user_input: 
    with st.spinner("Thinking..."): 
        response = chat_with_openrouter(user_input) 
        st.markdown("🤖 Bot: " + response) 
        speak(response)

# Mood trend graph
if os.path.exists("mood_log.csv"): 
    st.markdown("### 📈 Mood Over Time") 
    df = pd.read_csv("mood_log.csv") 
    df['timestamp'] = pd.to_datetime(df['timestamp']) 
    df = df.sort_values("timestamp") 
    st.line_chart(df.set_index("timestamp")["sentiment"])
