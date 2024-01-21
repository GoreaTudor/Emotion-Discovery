import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from nltk.stem import PorterStemmer
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
 
def preprocess_data(data, tokenizer):
    new_data = []
    for index, row in data.iterrows():
        test_split = row['text'].split()
        stemmed_words2 = [stemmer.stem(word) for word in test_split]
        token_list = tokenizer.texts_to_sequences([stemmed_words2])[0]
        new_data.append([token_list, row['label']])
    return new_data
 
def get_text(text):
    tokenizer3 = Tokenizer()
    tokenizer3.fit_on_texts(text)
    word_index3 = tokenizer3.word_index
 
    stemmed_wordss = [stemmer.stem(word) for word in word_index3.keys()]
 
    tokens_list = tokenizer2.texts_to_sequences([stemmed_wordss])[0]
 
    for i in range(len(tokens_list)):
        for j in range(length_of_longest_sentence - len(tokens_list)):
            tokens_list.append(0)
    return tokens_list
 
def get_text_input():
    user_input = input("Enter a sentence: ")
    return user_input
 
val_data = pd.read_csv("datasets\\formatted_validating.csv")
train_data = pd.read_csv("datasets\\formatted_training.csv")
test_data = pd.read_csv("datasets\\formatted_testing.csv")
 
#print("Validation data :",val_data.shape)
#print("Train data :",train_data.shape)
#print("Test data :",test_data.shape)
 
 
labels_dict = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise', 6: 'disgust', 7: 'neutral'}
train_data['label_name'] = train_data['label'].map(labels_dict)
#print(train_data.head())
 
#print(train_data.groupby(["label_name","label"]).size())
 
train_data["label_name"].value_counts().plot(kind='bar',color=['yellow', '#0c0d49', '#b82f2f', '#331e1e', 'red','#00fff7'])
# plt.show()
 
#print(train_data.isnull().sum())
#print(val_data.isnull().sum())
#print(test_data.isnull().sum())
 
all_list = train_data['text'].tolist() + test_data['text'].tolist() + val_data['text'].tolist()
 
tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(all_list)
word_index1 = tokenizer1.word_index
 
#print("Number of words without Stemming:", len(word_index1))
 
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in word_index1.keys()]
 
tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(stemmed_words)
word_index2 = tokenizer2.word_index
 
#print("Number of words with Stemming:", len(word_index2))
 
new_train_data = preprocess_data(train_data, tokenizer2)
#print(train_data['text'][0])
#print(new_train_data[0])
 
new_val_data = preprocess_data(val_data, tokenizer2)
#print(val_data['text'][0])
#print(new_val_data[0])
 
# Splitting into train_X and train_y
train_X = [row[0] for row in new_train_data]
train_y = [row[1] for row in new_train_data]
 
# Print the results
#print("train_X:", train_X[0])
#print("train_y:", train_y[0])
 
val_X = [row[0] for row in new_val_data]
val_y = [row[1] for row in new_val_data]
 
#print("val_X:", val_X[0])
#print("val_y:", val_y[0])
 
length_of_longest_sentence = len(max(train_X, key=len))
print(length_of_longest_sentence)
#print(length_of_longest_sentence)
#print(max(train_X, key=len))
 
for i in range(len(train_X)):
    for j in range(length_of_longest_sentence - len(train_X[i])):
        train_X[i].append(0)
 
for i in range(len(val_X)):
    for j in range(length_of_longest_sentence - len(val_X[i])):
        val_X[i].append(0)
 
train_X = np.array(train_X)
train_y = np.array(train_y)
val_X = np.array(val_X)
val_y = np.array(val_y)
 
#print(train_X.shape, train_y.shape)
#print(val_X.shape, val_y.shape)
 
# Convert labels to one-hot encoding
train_y_one_hot = to_categorical(train_y, num_classes=16000)
val_y_one_hot = to_categorical(val_y, num_classes=16000)
 
# User choice: Train or Test
user_choice = input("Enter 'train' to train the model or 'test' to test the model: ")
 
 
if user_choice.lower() == 'train':
 
    print("Number of epochs: ")
    input_epochs = int(input())
    # Training the model
    model = Sequential()
    model.add(Embedding(16000, 100, input_length=length_of_longest_sentence))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(16000, activation='softmax'))
    adam = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(train_X, train_y_one_hot, epochs=input_epochs, verbose=1, validation_data=(val_X, val_y_one_hot))
    print(model)
    # Plotting the training history
    plt.figure(figsize=(12, 6))
 
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
 
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
 
    plt.tight_layout()
    plt.show()
 
    last_accuracy = "{:.3f}".format(history.history['accuracy'][-1])
    print("Training Accuracy:", last_accuracy)
 
    # Save the trained model
    model.save("your_model.h5")
    print("Model has been saved.")
 
elif user_choice.lower() == 'test':
    # Load the saved model
    loaded_model = load_model("your_model.h5")
    print("Model has been loaded.")
 
    ok = 1
    while ok == 1:
        user_text = get_text_input()
        user_tokens = get_text([user_text])
        user_tokens = np.array(user_tokens)
        user_tokens = user_tokens.reshape(1, len(user_tokens))
 
        # Make predictions using the loaded model
        user_predictions = loaded_model.predict(user_tokens)
        user_predicted_class = np.argmax(user_predictions)
        print("User Input: " + user_text + "\n")
        print("The emotion in this sentence is:", user_predicted_class, labels_dict.get(user_predicted_class))
        print("Want to continue? Enter 1 for yes, 0 for no.")
        ok = int(input())
else:
    print("Invalid choice. Please enter 'train' or 'test'.")