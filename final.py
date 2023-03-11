import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import spacy
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import SimpleRNN
# from keras.layers.experimental.preprocessing import TextVectorization
from textblob import TextBlob
# Download the stopwords resource
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

# Load the transcription data
data = pd.read_csv('C:/Users/banda/OneDrive/Desktop/hackathon/transcritption.csv',header=None, names=['transcription_text'])
# print(data.iloc[2])  # prints the third row of the DataFrame
# Preprocess the text data
def preprocess_text(text):
    # Perform any necessary text cleaning
    # Convert the text to lowercase
    # Tokenize the text into words
    # Remove stop words
    # Perform stemming or lemmatization
     # Remove any URLs
    text = re.sub(r'http\S+', '', text)
    # Convert the text to lowercase
    text = text.lower()
    # Remove any punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # # Tokenize the text into words
    # words = text.split()
    # # Remove stop words
    # stop_words = set(stopwords.words('english'))
    # words = [word for word in words if word not in stop_words]
     # Tokenize the text using Spacy
    doc = nlp(text)
    words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a string
    processed_text = ' '.join(words)
    print(processed_text)
    return processed_text
    # return processed_text

preprocessed_text = [preprocess_text(text) for text in data['transcription_text']]

# Convert the text data to sequences of integer tokens
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_text)
sequences = tokenizer.texts_to_sequences(preprocessed_text)

# Pad the sequences to a fixed length
max_length = 50
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
print(type(padded_sequences))
# # Convert the severity labels to numerical values
# severity_labels = {'low': 0, 'medium': 1, 'high': 2}
# labels = [severity_labels[label] for label in data['severity']]

#sentiment analysis for assigning the labels as severity level
sentiments = [TextBlob(text).sentiment.polarity for text in preprocessed_text]

# Define severity labels based on sentiment scores
SEVERE = 'Severe'
MODERATE = 'Moderate'
MILD = 'Mild'
HAPPY = 'Happy'

# Map sentiment scores to severity labels
severity_labels = []
for score in sentiments:
    if score < 0.0:
        print(score)
        severity_labels.append(SEVERE)
    elif score >= 0.5:
        print(score)
        severity_labels.append(MILD)
    elif score == 0.0:
        severity_labels.append(HAPPY)
       
    else:
        print(score)
        severity_labels.append(MODERATE)
       
print(severity_labels)
# Convert severity labels to a numpy array
severity_labels = np.array(severity_labels)
print(type(severity_labels))
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


X_train, X_test, y_train, y_test = train_test_split(padded_sequences, severity_labels, test_size=0.2, random_state=42)
print(type(X_train))
print(type(X_test))
print(type(y_train))
print(type(y_test))

# Encode the severity labels as integers
label_encoder = LabelEncoder()

# Combine training and test sets
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))

label_encoder.fit(y)
# Transform labels in both sets
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# y_train = label_encoder.fit_transform(y_train)
# y_test = label_encoder.transform(y_test)
# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_length),
    # tf.keras.layers.Flatten(),
    # SimpleRNN(64),
    Bidirectional(LSTM(64)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_length),
#     Bidirectional(LSTM(64)),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(3, activation='softmax')
# ])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)




# Make predictions on new data
new_transcriptions = ["Customer: I'm sorry upset about my overdraft fees. Colleague: Let's review your account and see what we can do.", 
                      "Customer: I'm confused about my interest rate. Colleague: Let me check your account and explain it to you.",
                      "I'm overwhelmed to hear that you are not having any issues with your account. glad to hear that.",
                      "Customer: I'm disappointed with the service I've received. Colleague: I'm sorry to hear that. Let me try to make it right."]

preprocessed_new_transcriptions = [preprocess_text(transcription) for transcription in new_transcriptions]
new_sequences = tokenizer.texts_to_sequences(preprocessed_new_transcriptions)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length, padding='post')

predictions = model.predict(new_padded_sequences)

for i in range(len(predictions)):
    predicted_label = list(severity_labels)[np.argmax(predictions[i])]
    print(f"Transcription: {new_transcriptions[i]}")
    print(f"Predicted severity: {predicted_label}\n")
     
