import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.models import Sequential

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='ansi')
    data['상황'] = data['상황'].str.lower()
    # print(data[:5])

    value_counts = data['상황'].value_counts()
    print(value_counts)

    data['상황'] = data['상황'].replace(['happiness', 'angry', 'anger', 'disgust', 'fear', 'neutral', 'sadness', 'sad', 'surprise', '0', '1'], [1, 0, 0, 0, 0, 0.5, 0, 0, 0.5, 0, 1])
    
    data = data.drop(data[data['상황'] == 0.5].index)
    
    # print('null 값 여부 :',data.isnull().values.any())
    # print(data[data['상황'] == 0.5])  # 0.5인 행 출력
    
    data.drop_duplicates(subset=['발화문'], inplace=True)
    
    X_data = data['발화문']
    y_data = data['상황']
    return X_data, y_data

def preprocess_text(X_train, X_test):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_encoded = tokenizer.texts_to_sequences(X_train)
    X_test_encoded = tokenizer.texts_to_sequences(X_test)
    
    max_len = max(len(sample) for sample in X_train_encoded)
    vocab_size = len(tokenizer.word_index) + 1
    
    X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len)
    X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len)
    
    return tokenizer, X_train_padded, X_test_padded, vocab_size, max_len

def build_model(vocab_size, embedding_dim, hidden_units):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(LSTM(hidden_units, return_sequences=True))
    model.add(LSTM(hidden_units))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model

def train_model(model, X_train, y_train, epochs, batch_size, validation_split):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy

if __name__ == "__main__":
    file_path = 'data.csv'
    X_data, y_data = load_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, stratify=y_data)
    # print(y_train.value_counts())
    # print(y_test.value_counts())
    
    tokenizer, X_train_padded, X_test_padded, vocab_size, max_len = preprocess_text(X_train, X_test)
    
    embedding_dim = 32
    hidden_units = 32
    model = build_model(vocab_size, embedding_dim, hidden_units)
    
    epochs = 10
    batch_size = 64
    validation_split = 0.2
    train_model(model, X_train_padded, y_train, epochs, batch_size, validation_split)
    
    while True:
        text = input("문장을 입력해주세요 ('exit' 또는 '0'을 입력하면 종료됩니다): ")
        if text.lower() == 'exit' or text == '0':
            break
        else:
            X_test_encoded = tokenizer.texts_to_sequences([text])
            X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len)
            prediction = model.predict(X_test_padded)[0][0]
            print(prediction)
            if prediction > 0.5:
                print("긍정")
            else:
                print("부정")
