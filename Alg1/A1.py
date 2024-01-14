import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from Loader import load_dataset

if __name__ == '__main__':
    dataset = load_dataset()
    train_set, val_set, test_set = dataset['train'], dataset['val'], dataset['test']

    # Conversion between labels and numeric values
    emotion_labels = {'anger': 0, 'sadness': 1, 'fear': 2, 'neutral': 3, 'joy': 4, 'surprise': 5, 'disgust': 6}

    train_set['emotions_numeric'] = np.array([emotion_labels[emotion] for emotion in train_set['emotions']])
    val_set['emotions_numeric'] = np.array([emotion_labels[emotion] for emotion in val_set['emotions']])
    test_set['emotions_numeric'] = np.array([emotion_labels[emotion] for emotion in test_set['emotions']])

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_set['utterances'])

    train_sequences = tokenizer.texts_to_sequences(train_set['utterances'])
    val_sequences = tokenizer.texts_to_sequences(val_set['utterances'])
    test_sequences = tokenizer.texts_to_sequences(test_set['utterances'])

    # Padding
    train_padded = pad_sequences(train_sequences)
    val_padded = pad_sequences(val_sequences)
    test_padded = pad_sequences(test_sequences)

    # Model Definition
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=train_padded.shape[1]))
    model.add(LSTM(128))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(len(emotion_labels), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training
    model.fit(train_padded, train_set['emotions_numeric'], epochs=5, batch_size=32)

    # Results
    test_loss, test_accuracy = model.evaluate(test_padded, test_set['emotions_numeric'])
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
