import streamlit as st
import os
from dotenv import load_dotenv
import requests
import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# Load environment variables (API key)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Download NLTK data (only first time)
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# System prompt defining the bot's personality and behavior
SYSTEM_PROMPT = (
    "You are a warm, empathetic, and non-judgmental mental health assistant. "
    "Your goal is to listen carefully and deeply understand the user's feelings and experiences. "
    "Respond with thoughtful, personalized, and supportive advice, just like a compassionate psychologist or counselor. "
    "Provide motivational words, practical mental wellness tips, coping strategies, and encouragement. "
    "Validate the user's feelings and gently guide them towards self-care and a positive mindset. "
    "Always be respectful, kind, and patient. "
    "Do not give medical or diagnostic advice. "
    "Make the conversation feel safe and supportive."
)

# Streamlit UI setup
st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("🧠 Mental Health Chatbot")
st.markdown("Talk about your feelings. I'm here to listen and support you anonymously. ❤️")

# Initialize chat history with system prompt if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# Daily mood check-in section
st.markdown("### ☀ Daily Mood Check-in")
user_mood = st.text_input("How are you feeling today? (Optional, press Enter to submit)")

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

# Chat input box
st.markdown("### 💬 Chat with me")
user_input = st.text_input("Type your message here and press Enter:")

def chat_with_openrouter(history):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mixtral-8x7b",  # Free and powerful model
        "messages": history
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# Handle new user input
if user_input:
    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        try:
            # Get AI response
            bot_response = chat_with_openrouter(st.session_state.chat_history)
            # Append bot response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        except Exception as e:
            st.error(f"Failed to connect to AI. Please check your API key or internet connection.\n{e}")
            bot_response = None

# Display the conversation
if "chat_history" in st.session_state:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "assistant":
            st.markdown(f"**Bot:** {message['content']}")

# Mood over time graph
if os.path.exists("mood_log.csv"):
    st.markdown("### 📈 Mood Over Time")
    df = pd.read_csv("mood_log.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")
    st.line_chart(df.set_index("timestamp")["sentiment"])

# Disclaimer for safety
st.markdown(
    "<small><i>Disclaimer: This chatbot is for supportive conversation only and does not replace professional mental health care. "
    "If you are in crisis or need urgent help, please contact a qualified mental health professional or emergency services.</i></small>",
    unsafe_allow_html=True
)