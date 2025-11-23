import streamlit as st
import pickle
import os
import numpy as np

try:
    from model import train_model
except Exception as e:
    st.error("model.py failed to import. Real error below:")
    st.exception(e)
    st.stop()
    
st.set_page_config(page_title="AI Stress Analyzer", page_icon="🧠")


if not os.path.exists("model.pkl"):
    train_model()

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
vectorizer = bundle["vectorizer"]
acc = bundle["accuracy"]

st.title("🧠 AI Stress Analyzer")
st.write("Type how you're feeling, and the model estimates your stress level in real time.")

st.caption(f"Model holdout accuracy: **{acc:.2f}** (tiny dataset — improves as you add data)")

text = st.text_area(
    "How are you feeling today?",
    placeholder="Example: I'm worried about my exams and can't focus..."
)

if st.button("Analyze Stress"):
    if not text.strip():
        st.warning("Please enter a short message.")
    else:
        X = vectorizer.transform([text])
        probs = model.predict_proba(X)[0]
        pred = int(np.argmax(probs))
        confidence = float(np.max(probs))

        st.subheader("Result")

        if pred == 0:
            st.success(f"🟢 Low Stress (confidence {confidence:.2f})")
            st.write("You seem relatively calm. Keep up healthy routines!")
            st.write("Suggestion: maintain sleep, hydration, and short breaks.")
        elif pred == 1:
            st.warning(f"🟡 Moderate Stress (confidence {confidence:.2f})")
            st.write("You’re feeling some pressure. Try to pace your workload.")
            st.write("Suggestion: break tasks into small steps and take a short reset.")
        else:
            st.error(f"🔴 High Stress (confidence {confidence:.2f})")
            st.write("Your text shows strong stress signals.")
            st.write("Suggestion: rest, talk to someone you trust, and reduce overload if possible.")

st.write("---")
st.caption("AI Stress Analyzer — NLP + Logistic Regression prototype (2025)")
