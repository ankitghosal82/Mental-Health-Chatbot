## 🧠 Anonymous Mental Health Chatbot (AI-Powered)

This is an anonymous, AI-powered **mental health chatbot** built using **Python** and **Streamlit**, connected to **OpenRouter’s free GPT-based API**. It allows users to express emotions freely, track their mood, and receive supportive AI responses with both voice input and output.

---

## 🔟 Key Features

- 💬 **GPT-Style Chatbot**  
  Uses OpenRouter API with models like `mistralai/mixtral-8x7b` to generate intelligent replies.

- 🎙️ **Voice Input**  
  Speak directly to the chatbot using your microphone via `SpeechRecognition`.

- 🔊 **Voice Output**  
  Get AI replies spoken back to you using the `pyttsx3` text-to-speech engine.

- 📅 **Daily Mood Check-In**  
  Prompted once a day to describe your mood — it’s stored for emotional tracking.

- 🧠 **Sentiment Analysis (VADER)**  
  Detects whether you’re feeling positive, neutral, or negative based on your input.

- 📊 **Mood Graph Visualization**  
  View your emotional history through automatically generated sentiment graphs.

- 🔐 **100% Anonymous & Local**  
  All data is stored locally. No personal info is uploaded or tracked.

- 🔧 **Easy API Integration via `.env`**  
  Keeps your API key private and secure using a `.env` file.

- 📁 **Mood Logging in CSV**  
  Saves your emotional entries in `mood_log.csv` for long-term analysis.

- 💻 **Lightweight, Fast, and Mobile-Friendly UI**  
  Streamlit makes the interface modern, simple, and fast to deploy.

---

## ✅ Conclusion

This project creates a **safe, private, and interactive space** where users can express their feelings, get AI-generated support, and track their mental health journey over time. Ideal for students, AI beginners, or developers looking to combine **NLP + voice tech + mental health awareness**.

> 🌟 Try it, improve it, and share it to promote mental well-being through tech!

---

## 📌 Tech Stack

- Python
- Streamlit
- OpenRouter API
- NLTK (VADER)
- pyttsx3
- SpeechRecognition
- Matplotlib
- Pandas
