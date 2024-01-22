import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def preprocess_data(data, tokenizer):

    new_data = [] #Initializes an empty list to store the preprocessed data.
    for index, row in data.iterrows(): #Iterates over each row in the input DataFrame `data`.
        test_split = row['text'].split()    #Splits the text in the current row into a list of words. This is done using the `split()` method, which by default splits the string at spaces.
        stemmed_words2 = [stemmer.stem(word) for word in test_split]    #Applies stemming to each word in the list of words obtained in the previous step.
        token_list = tokenizer.texts_to_sequences([stemmed_words2])[0]  #Converts the list of stemmed words into a sequence of tokens using the tokenizer (`tokenizer2`). The resulting `token_list` is a list of integer indices corresponding to the words in the vocabulary.
        new_data.append([token_list, row['label']]) #Appends the `token_list` and the label in the current row to the list `new_data`.
    return new_data

def get_text(text): #Defines a function that takes a sentence as input and returns a list of tokens corresponding to the words in the sentence.

    tokenizer3 = Tokenizer()    #Initializes a new tokenizer.
    tokenizer3.fit_on_texts(text)   #Fits the tokenizer on the input text.
    word_index3 = tokenizer3.word_index  #Creates a dictionary mapping each word in the vocabulary to a unique integer index.
    stemmed_wordss = [stemmer.stem(word) for word in word_index3.keys()]    #Applies stemming to each word in the vocabulary.
    tokens_list = tokenizer2.texts_to_sequences([stemmed_wordss])[0]    #Converts the list of stemmed words into a sequence of tokens using the tokenizer (`tokenizer2`). The resulting `tokens_list` is a list of integer indices corresponding to the words in the vocabulary.

    for i in range(len(tokens_list)):   #Iterates over each token in the `tokens_list`.
        for j in range(length_of_longest_sentence - len(tokens_list)):  #Iterates over each position in the `tokens_list` that is empty.
            tokens_list.append(0)   #Appends a 0 to the `tokens_list` at the current position.
    return tokens_list

def get_text_input():
    user_input = input("Enter a sentence: ")
    return user_input


val_data = pd.read_csv("datasets\\formatted_validating.csv")
train_data = pd.read_csv("datasets\\formatted_training.csv")
test_data = pd.read_csv("datasets\\formatted_testing.csv")

labels_dict = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise', 6: 'disgust', 7: 'neutral'}
train_data['label_name'] = train_data['label'].map(labels_dict)

all_list = train_data['text'].tolist() + test_data['text'].tolist() + val_data['text'].tolist()

tokenizer1 = Tokenizer()    #Initializes a new tokenizer.
tokenizer1.fit_on_texts(all_list)   #Fits the tokenizer on the input text.
word_index1 = tokenizer1.word_index #Creates a dictionary mapping each word in the vocabulary to a unique integer index.

# the number of unique words in the vocabulary
vocab_size = len(word_index1) + 1
print(vocab_size)

stemmer = PorterStemmer()   #Initializes a new stemmer.
stemmed_words = [stemmer.stem(word) for word in word_index1.keys()]   #Applies stemming to each word in the vocabulary.

tokenizer2 = Tokenizer()    #Initializes a new tokenizer.
tokenizer2.fit_on_texts(stemmed_words)  #Fits the tokenizer on the stemmed words.
word_index2 = tokenizer2.word_index #Creates a dictionary mapping each word in the vocabulary to a unique integer index.


new_train_data = preprocess_data(train_data, tokenizer2)    #Preprocesses the training data.

new_val_data = preprocess_data(val_data, tokenizer2)    #Preprocesses the validation data.


# Splitting into train_X and train_y
train_X = [row[0] for row in new_train_data]    #Creates a list of token sequences corresponding to the tokenized sentences in the training data.
train_y = [row[1] for row in new_train_data]    #Creates a list of labels corresponding to the tokenized sentences in the training data.


val_X = [row[0] for row in new_val_data]    #Creates a list of token sequences corresponding to the tokenized sentences in the validation data.
val_y = [row[1] for row in new_val_data]    #Creates a list of labels corresponding to the tokenized sentences in the validation data.

length_of_longest_sentence = len(max(train_X, key=len))
print(length_of_longest_sentence)

for i in range(len(train_X)):   #Iterates over each token sequence in the training data.
    for j in range(length_of_longest_sentence - len(train_X[i])):   #Iterates over each position in the token sequence that is empty.
        train_X[i].append(0)    #Appends a 0 to the token sequence at the current position.

for i in range(len(val_X)):
    for j in range(length_of_longest_sentence - len(val_X[i])):
        val_X[i].append(0)

train_X = np.array(train_X) #Converts the list of token sequences into a numpy array.
train_y = np.array(train_y) #Converts the list of labels into a numpy array.
val_X = np.array(val_X) #Converts the list of token sequences into a numpy array.
val_y = np.array(val_y) #Converts the list of labels into a numpy array.


# Convert labels to one-hot encoding
train_y_one_hot = to_categorical(train_y, num_classes=8)    #Converts the training labels to one-hot encoding.
val_y_one_hot = to_categorical(val_y, num_classes=8)    #Converts the validation labels to one-hot encoding.

# User choice: Train or Test
user_choice = input("Enter 'train' to train the model or 'test' to test the model: ")


if user_choice.lower() == 'train':

    print("Number of epochs: ")
    input_epochs = int(input())
    # Training the model
    model = Sequential()
    model.add(Embedding(6398, 100, input_length=length_of_longest_sentence))   #Creates an embedding layer with 16000 as the input dimension, 100 as the output dimension, and length_of_longest_sentence as the input length.    model.add(Bidirectional(LSTM(150))) #Creates a bidirectional LSTM layer with 150 units.
    model.add(Dense(6398, activation='softmax'))   #Creates a dense layer with 16000 units and softmax activation.
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