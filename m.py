import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from random import choice

nltk.download('punkt')

def train_model():
    data = {
        "sad": [
            "I feel down",
            "I'm so depressed",
            "Nothing makes me happy",
            "Feeling blue",
            "I am overwhelmed with sadness"
        ],
        "happy": [
            "I feel great",
            "I'm very happy",
            "Life is amazing",
            "Feeling joyful",
            "Everything is going well"
        ],
        "angry": [
            "I hate this",
            "I'm furious",
            "So much rage in me",
            "I am really upset",
            "Feeling angry"
        ],
        "anxious": [
            "I'm nervous",
            "Feeling anxious",
            "My heart is racing",
            "I can't calm down",
            "Worried about everything"
        ]
    }

    texts, labels = [], []
    for mood, samples in data.items():
        for sentence in samples:
            texts.append(sentence)
            labels.append(mood)

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(texts, labels)
    return model

# More detailed, human-like responses with tips and motivations
responses = {
    "sad": [
        "I'm sorry you're feeling this way. Sometimes, talking about what's bothering you helps. Remember, it's okay to ask for help.",
        "It's normal to feel down sometimes. Try to do something small today that brings you joy, like listening to your favorite song or taking a short walk.",
        "Remember, your feelings are valid. If you want, you can try writing down your thoughts—it can help lighten the emotional load.",
        "You're not alone in this. Reaching out to a friend or family member might help you feel supported.",
        "When sadness feels overwhelming, deep breathing or mindfulness can bring a little peace. Want me to guide you through a breathing exercise?"
    ],
    "happy": [
        "That's wonderful to hear! Celebrate these moments and savor the good feelings.",
        "Happiness is a beautiful thing. Keep doing what brings you joy and spread that positive energy!",
        "Glad you're feeling good! Maybe try sharing your happiness with someone else—it can make your day even better.",
        "It's great to see you smiling today. Keeping a gratitude journal might help keep this positive vibe going!",
        "Enjoy these happy moments—they're important for your overall well-being."
    ],
    "angry": [
        "Anger is a strong emotion. Try taking a few deep breaths and count to ten before reacting—it really helps.",
        "It's okay to feel angry. Sometimes channeling that energy into a physical activity like jogging or punching a pillow can release tension.",
        "Remember, you're in control of your emotions. If you'd like, I can share some quick relaxation techniques.",
        "Try to pause and identify what exactly triggered your anger. Understanding it can help you manage it better.",
        "Expressing anger calmly and constructively is powerful. If you want, you can write down what you're feeling before speaking out."
    ],
    "anxious": [
        "Anxiety can be tough, but you’re not alone. Let's try some slow, deep breathing together to calm your mind.",
        "Try focusing on the present moment. You might want to try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
        "Remember, it's okay to take breaks and step back when anxiety builds up.",
        "Physical exercise, even a short walk, can really help ease anxious feelings.",
        "You’re doing your best, and that’s enough. Would you like me to share a calming meditation audio or exercise?"
    ],
    "default": [
        "I'm here to listen. Tell me more about how you're feeling.",
        "Thanks for sharing. Would you like some tips to help you feel better?",
        "Your feelings matter. I'm here whenever you want to talk.",
        "Sometimes just putting feelings into words can help. Feel free to share.",
        "No matter what you’re feeling, it’s okay. Let’s take it one step at a time."
    ]
}

def get_bot_response(user_input, model):
    mood = model.predict([user_input])[0]
    reply_list = responses.get(mood, responses["default"])
    return mood.capitalize(), choice(reply_list)

# Train model on app start
model = train_model()

# Streamlit UI
st.set_page_config(page_title="MindEase", layout="centered")
st.title("🧠 MindEase – Mental Health ChatBot")
st.write("Your anonymous mental health companion 💙")
st.markdown("---")

user_input = st.text_input("How are you feeling today?", placeholder="Type here...")

if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please type something.")
    else:
        mood, reply = get_bot_response(user_input, model)
        st.success(f"**Detected Mood:** `{mood}`")
        st.markdown(f"**Bot Response:** {reply}")

