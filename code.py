import nltk
from tkinter import *
from tkinter import ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np

# NLTK download (run once)
nltk.download('punkt')
nltk.download('stopwords')

# 🧠 Text Summarizer Function
def summarize_text():
    text = text_input.get("1.0", END).strip()
    if not text:
        text_output.delete("1.0", END)
        text_output.insert(END, "❗ Please enter text.")
        return

    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        text_output.delete("1.0", END)
        text_output.insert(END, text)
        return

    stop_words = stopwords.words('english')
    tfidf = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = tfidf.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    scores = similarity_matrix.sum(axis=1)

    ranked = np.argsort(scores)[-3:]
    ranked = sorted(ranked)
    summary = " ".join([sentences[i] for i in ranked])

    text_output.delete("1.0", END)
    text_output.insert(END, summary)

# ✨ Clear Function
def clear_text():
    text_input.delete("1.0", END)
    text_output.delete("1.0", END)

# 🎨 Main GUI Window
root = Tk()
root.title("✨ Smart Text Summarizer")
root.geometry("780x560")
root.configure(bg="#f2f5f7")

# 📝 Fonts
TITLE_FONT = ("Helvetica", 18, "bold")
LABEL_FONT = ("Helvetica", 12, "bold")
TEXT_FONT = ("Segoe UI", 11)

# 🟦 Title
title_label = Label(root, text="AI-Powered Text Summarizer", font=TITLE_FONT, bg="#f2f5f7", fg="#2c3e50")
title_label.pack(pady=15)

# 🔷 Input Label
Label(root, text="Enter Paragraph Below:", font=LABEL_FONT, bg="#f2f5f7", fg="#2c3e50").pack(anchor="w", padx=30)
text_input = Text(root, height=10, width=85, font=TEXT_FONT, bd=2, relief=GROOVE, bg="#ffffff")
text_input.pack(padx=30, pady=5)

# ▶ Buttons Frame
btn_frame = Frame(root, bg="#f2f5f7")
btn_frame.pack(pady=10)

summarize_btn = Button(btn_frame, text="🔍 Summarize", font=LABEL_FONT, bg="#27ae60", fg="white", width=15, command=summarize_text)
summarize_btn.grid(row=0, column=0, padx=10)

clear_btn = Button(btn_frame, text="🧹 Clear", font=LABEL_FONT, bg="#c0392b", fg="white", width=10, command=clear_text)
clear_btn.grid(row=0, column=1, padx=10)

# 🟩 Output Label
Label(root, text="Summary:", font=LABEL_FONT, bg="#f2f5f7", fg="#2c3e50").pack(anchor="w", padx=30)
text_output = Text(root, height=6, width=85, font=TEXT_FONT, bd=2, relief=GROOVE, bg="#f9f9f9")
text_output.pack(padx=30, pady=5)

# 🟦 Run GUI
root.mainloop()
