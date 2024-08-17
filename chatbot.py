import random
import json
import pickle
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# Load the "intents.json" file
try:
    with open("intents.json", "r") as f:
      intents = json.load(f)
    print("File loaded successfully!")
except FileNotFoundError:
    print("The file 'intents.json' was not found. Make sure it exists in the current directory.")
except json.JSONDecodeError:
    print("The file 'intents.json' does not contain valid JSON data. Check the file's contents.")

# Lemmatizer to get the roots (Lemma) of any word
lemmatizer = WordNetLemmatizer()

# Load the files "words" and "classes" which are generated in the "Medical_Assistant_AI_Chatbot.ipynb"
# Module "pickle" is used here to download and load "words" and "classes" files from "Medical_Assistant_AI_Chatbot.ipynb"
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Use load_model from tensorflow.keras to load the pre-trained model
# Full code for Neural Network (model) implementation is in the "Medical_Assistant_AI_Chatbot.ipynb" file. This model was trained in "Medical_Assistant_AI_Chatbot.ipynb" (Google Colab) for faster execution using cloud GPU
model = load_model("medical_assistant_AI_chatbot.keras")


def clean_sentence(sentence):
  # Tokenize the sentence into words
  sentence_words = nltk.word_tokenize(sentence)
  # Lemmatize each word
  sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
  return sentence_words

def bag_of_words(sentence):
  # Clean and tokenize the sentence
  sentence_words = clean_sentence(sentence)
  bag = [0] * len(words)
  # Mark the presence of words in the bag (1)
  for w in sentence_words:
    for i, word in enumerate(words):
      if word == w:
        bag[i] = 1
  
  return np.array(bag)

def predict_class(sentence):
  # Convert sentence to bag of words
  bow = bag_of_words(sentence)
  # Predict class probabilities
  prediction = model.predict(np.array([bow]))[0]
  # ERROR_THRESHOLD = 0.25
  # Filter out predictions below the error threshold
  results = [[i, r] for i, r in enumerate(prediction)]
  # Sort results by probability
  results.sort(key=lambda x: x[1], reverse=True)
  
  return_list = []
  for r in results:
    # Map class index to class name
    return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
  return return_list

def get_response(intents_list, intents_json):
  # Get the intent with the highest probability
  tag = intents_list[0]["intent"]
  all_intents = intents_json["intents"]
  for i in all_intents:
    if i["tag"] == tag:
      # Select a random response for the identified intent
      result = random.choice(i["responses"])
      break
  return result


print("Your Medical Assitant is ready! Type 'quit' to exit the bot")

while True:
  message = input("You: ")
  if message.lower() == "quit":
    break
  # Predict the class of the input message
  ints = predict_class(message)
  # Get a response based on the predicted class
  response = get_response(ints, intents)
  print(f"Medical Assistant: {response}")