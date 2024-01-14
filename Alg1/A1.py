import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from Loader import load_dataset

if __name__ == '__main__':
    train, validation, test = load_dataset()

    # Tokenization and padding example using TensorFlow's Tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train.utterances)
    utterances_sequences = tokenizer.texts_to_sequences(train.utterances)
    utterances_padded = tf.keras.preprocessing.sequence.pad_sequences(utterances_sequences)

    # Convert emotion labels to numerical values
    emotion_labels = {'neutral': 0, 'sad': 1, 'surprised': 2, 'happy': 3}  # Add more as needed
    emotions_numeric = np.array([emotion_labels[emotion] for emotion in train.emotions])

    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=utterances_padded.shape[1]))
    model.add(LSTM(128))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(emotion_labels), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(utterances_padded, emotions_numeric, epochs=10, validation_split=0.2)
