import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

# Predefined questions and answers
questions = ["hi", "how are you", "what's your name", "tell me a joke", "what is AI", "what is Python",
             "tell me about machine learning","who created Python","what is deep learning","what is NLP",
             "what is your purpose","what is data science","who is Alan Turing","what is the Turing Test",
               ]
answers = ["Hi!", "I'm doing well, thank you!", "I'm PyBot. Ask me a math question, please.", 
           "Why don't scientists trust atoms? Because they make up everything!", 
           "AI stands for Artificial Intelligence.", "Python is a programming language."]

# Preprocess questions to remove stop words and lemmatize
def preprocess_question(question):
    doc = nlp(question.lower())
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return " ".join(tokens) if tokens else question

# Compute vector for questions
def vectorize(questions):
    vectors = []
    for question in questions:
        vectors.append(nlp(question).vector)
    return np.array(vectors)

# Preprocess and vectorize predefined questions
preprocessed_questions = [preprocess_question(q) for q in questions]
question_vectors = vectorize(preprocessed_questions)

def get_response(user_input):
    # Preprocess the user input
    preprocessed_input = preprocess_question(user_input)
    print(f"Preprocessed '{user_input}' to '{preprocessed_input}'")

    # Vectorize the user input
    input_vector = nlp(preprocessed_input).vector.reshape(1, -1)
    print(f"Vector for '{preprocessed_input}': {input_vector}")

    # Compute cosine similarity between user input and predefined questions
    similarities = cosine_similarity(input_vector, question_vectors)
    print(f"Similarities: {similarities}")

    # Find the most similar predefined question
    most_similar_index = np.argmax(similarities)
    print(f"Most similar question index: {most_similar_index} with similarity {similarities[0][most_similar_index]}")

    # Return the answer if similarity is above a threshold
    if similarities[0][most_similar_index] > 0.5:
        return answers[most_similar_index]
    else:
        return "I'm not sure I understand. Can you rephrase?"

def send_message():
    user_input = user_entry.get()
    if user_input.lower() in ["exit", "quit"]:
        root.quit()
    else:
        response = get_response(user_input)
        chat_area.config(state=tk.NORMAL)
        chat_area.insert(tk.END, f"You: {user_input}\n")
        chat_area.insert(tk.END, f"Bot: {response}\n")
        chat_area.config(state=tk.DISABLED)
        user_entry.delete(0, tk.END)

# Initialize GUI
root = tk.Tk()
root.title("Chatbot")

chat_area = scrolledtext.ScrolledText(root, state='disabled', wrap='word', width=50, height=20)
chat_area.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

user_entry = tk.Entry(root, width=40)
user_entry.grid(row=1, column=0, padx=10, pady=10)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.grid(row=1, column=1, padx=10, pady=10)

root.mainloop()
