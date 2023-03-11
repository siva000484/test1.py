import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

# Pad the sequences to a fixed length
maxlen = 500
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=32, input_length=maxlen))
model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the saved weights
model.load_weights('C:/Users/banda/OneDrive/Desktop/hackathon/my_model_weights.h5')

# Use the model to make predictions
y_pred = model.predict(x_test)
print(y_pred)

# Create some test data
test_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# Make predictions on the test data
predictions = model.predict(test_data)

# Print the predictions
print(predictions)
