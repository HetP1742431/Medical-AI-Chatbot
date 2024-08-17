# 🏥 Medical AI Chatbot
The Medical AI Chatbot is a machine learning-based application designed to provide users with instant medical information and assistance. By leveraging advanced natural language processing (NLP) techniques and deep learning models, this chatbot can understand and respond to a wide range of medical-related queries with accuracy and relevance.

## 🛠 Core Features
- **Natural Language Processing (NLP)**:
  - Implemented using NLTK, the chatbot preprocesses and lemmatizes user input to accurately interpret and respond to queries.
  - The text data is tokenized and structured, allowing the chatbot to process the information efficiently and understand the intent behind user queries.

- **Intent Classification**:
  - The chatbot utilizes a custom `intents.json` dataset, which categorizes medical conditions, symptoms, and related queries into specific intents such as "Diabetes" and "Flu."
  - Text data is converted into numerical vectors using Bag of Words (BoW) and TF-IDF techniques, facilitating precise intent classification by the deep learning model.

- **Deep Learning Model**:
  - Built with TensorFlow and Keras, the model is trained to classify user intents based on their queries.
  - The model architecture includes dense layers with ReLU activation and a softmax layer for multi-class classification, ensuring accurate and contextually relevant responses.

- **Model Deployment**:
  - The trained model is saved and deployed using Keras, allowing for real-time predictions within the chatbot application.
  - The chatbot interacts with users through a command-line interface (CLI), providing instant medical advice based on the predicted intent.

## 🛠 Tech Stack
- **Natural Language Processing**: NLTK
- **Machine Learning**: TensorFlow, Keras
- **Data Manipulation**: NumPy, Pandas
- **Data Storage**: JSON
- **Model Deployment**: Python, Pickle

## 📋 How to Run the Project Locally

### Prerequisites
- Python 3.6 or higher
- Pip package manager

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/HetP1742431/Medical-AI-Chatbot.git
   cd Medical-AI-Chatbot
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
3. Run the chatbot:
   ```bash
   python chatbot.py

## Usage:
- The chatbot will prompt you to enter a medical-related query (e.g., "What should I do if I have a headache?").
- Based on the input, the chatbot will respond with relevant medical advice using the trained model to predict the intent and provide an appropriate response.

## What I Learned
- Developed expertise in NLP techniques, particularly in text preprocessing, tokenization, and lemmatization.
- Gained experience in building and training deep learning models for intent classification using TensorFlow and Keras.
- Improved my understanding of deploying machine learning models into real-time applications and integrating them with user interfaces.
- Enhanced problem-solving skills through the challenges of debugging and optimizing NLP pipelines and machine learning models.

This project significantly contributed to my growth as a software engineer and machine learning practitioner, equipping me with the skills to develop and deploy intelligent systems that meet real-world needs.

- **Live Demo Video**: [Demo video](https://drive.google.com/file/d/1IJ9xREWAgR-p1avH_TZ_i5ZayEMxd5zk/view?usp=drive_link).
