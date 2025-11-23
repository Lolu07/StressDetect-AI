# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def get_training_data():
    
    data = [
        # LOW stress examples (0)
        ("I feel calm and relaxed today", 0),
        ("I had a good sleep and feel fine", 0),
        ("Everything is under control and going smoothly", 0),
        ("I'm feeling positive and focused", 0),
        ("Today was productive and not too stressful", 0),
        ("I feel steady and okay emotionally", 0),
        ("I'm enjoying my work and not overwhelmed", 0),
        ("I took breaks and feel balanced", 0),

        # MEDIUM stress examples (1)
        ("I'm a bit tired and have a lot to do", 1),
        ("I feel slightly anxious about deadlines", 1),
        ("I'm stressed but managing so far", 1),
        ("I have been busy and feeling pressure", 1),
        ("I'm worried about upcoming exams", 1),
        ("My mind feels a bit heavy today", 1),
        ("I'm restless but still functioning", 1),
        ("I feel tension but not breaking down", 1),

        # HIGH stress examples (2)
        ("I feel overwhelmed and can't focus", 2),
        ("I'm exhausted and everything feels too much", 2),
        ("I have panic about my workload", 2),
        ("I feel burnt out and mentally drained", 2),
        ("I can't keep up with everything anymore", 2),
        ("My stress is out of control and I feel hopeless", 2),
        ("I feel anxious all the time and can't sleep", 2),
        ("I'm breaking down from pressure", 2),
    ]
    return pd.DataFrame(data, columns=["text", "label"])

def train_model():
    df = get_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2) 
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        multi_class="ovr"
    )

    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    
    with open("model.pkl", "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer, "accuracy": acc}, f)

    print(f"Model trained. Holdout accuracy: {acc:.2f}")

if __name__ == "__main__":
    train_model()
