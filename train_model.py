import csv
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import tkinter as tk
from keras.metrics import Precision, Recall
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization, Bidirectional, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from datetime import datetime
from utils.text_cleaning import clean_text
from utils.file_helpers import save_pickle
from utils.embedding_helpers import load_pretrained_embeddings
from utils.tokenizer_helpers import calculate_optimal_num_words
from utils.preprocessing_helpers import load_preprocessed_data, save_preprocessed_data
from keras.optimizers import Adam

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funzione per configurare la GPU per l'uso con TensorFlow
def configure_gpu(console=None):
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            message = f'{len(gpus)} GPU(s) configurate per l\'uso.'
        else:
            message = 'Nessuna GPU disponibile. Utilizzando la CPU.'
        logging.info(message)
        if console:
            console.put( message + '\n')
    except RuntimeError as e:
        message = f'Errore nella configurazione delle GPU: {e}'
        logging.error(message)
        if console:
            console.put( message + '\n')

# Funzione per caricare e preprocessare i dati
def load_and_preprocess_data(file_path, console=None):
    logging.info("Caricamento dei dati di addestramento...")
    if console:
        console.put( "Caricamento dei dati di addestramento...\n")
    try:
        df_train = pd.read_csv(file_path, encoding='iso-8859-1')
    except Exception as e:
        message = f"Errore nel caricamento dei dati: {e}"
        logging.error(message)
        if console:
            console.put( message + '\n')
        return None
    
    logging.info("Mappatura dei sentimenti...")
    if console:
        console.put( "Mappatura dei sentimenti...\n")
    sentiment_mapping = {'POSITIVE': 1, 'NEGATIVE': 0}
    df_train['scoreSentiment'] = df_train['scoreSentiment'].map(sentiment_mapping)
    
    logging.info("Pulizia del testo delle recensioni di addestramento...")
    if console:
        console.put( "Pulizia del testo delle recensioni di addestramento...\n")
    df_train['reviewText'] = df_train['reviewText'].apply(clean_text)
    df_train = df_train[df_train['reviewText'].str.strip() != ""]
    
    return df_train

# Funzione per tokenizzare le recensioni
def tokenize_reviews(df_train, optimal_num_words, console=None):
    reviews = df_train['reviewText'].tolist()
    labels = df_train['scoreSentiment'].tolist()

    logging.info("Tokenizzazione delle recensioni...")
    if console:
        console.put( "Tokenizzazione delle recensioni...\n")
    
    tokenizer = Tokenizer(num_words=optimal_num_words)
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    clean_sequences = [seq for seq in sequences if seq]
    clean_labels = [label for seq, label in zip(sequences, labels) if seq]
    
    return tokenizer, clean_sequences, clean_labels

# Funzione per fare il padding delle sequenze e dividere i dati
def pad_sequences_and_split_data(clean_sequences, clean_labels, padding_strategy='percentile', padding_value=0, console=None):
    logging.info("Calcolo della lunghezza delle sequenze...")
    if console:
        console.put( "Calcolo della lunghezza delle sequenze...\n")
    lengths = [len(seq) for seq in clean_sequences]

    if padding_strategy == 'mean_std':
        mean_len = int(np.mean(lengths))
        std_len = int(np.std(lengths))
        max_len = mean_len + std_len
    elif padding_strategy == 'percentile':
        max_len = int(np.percentile(lengths, 95))
    else:
        max_len = max(lengths)

    logging.info(f"Padding delle sequenze alla lunghezza massima di {max_len}...")
    if console:
        console.put( f"Padding delle sequenze alla lunghezza massima di {max_len}...\n")
    data_padded = pad_sequences(clean_sequences, maxlen=max_len, padding='post', truncating='post', value=padding_value)
    labels = np.array(clean_labels)

    logging.info("Suddivisione dei dati in set di addestramento, validazione e test...")
    if console:
        console.put( "Suddivisione dei dati in set di addestramento, validazione e test...\n")
    
    X_train, X_rem, y_train, y_rem = train_test_split(data_padded, labels, test_size=0.25, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.52, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, max_len

# Funzione per costruire il modello di rete neurale
def build_model(embedding_matrix, max_len, embedding_dim, console=None):
    logging.info("Creazione del modello...")
    if console:
        console.put( "Creazione del modello...\n")
    
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())  
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())  
    model.add(Dropout(0.40))
    model.add(Dense(1, activation='sigmoid'))
    
    logging.info("Compilazione del modello...")
    if console:
        console.put( "Compilazione del modello...\n")
    model.compile(optimizer=Adam(learning_rate=0.0007), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
    
    return model

# Funzione per addestrare il modello
def train_model(model, X_train, y_train, X_test, y_test, console=None):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = f'models/sentiment_analysis_{timestamp}_epoch_{{epoch:02d}}_val_loss_{{val_loss:.2f}}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        checkpoint
    ]
    
    logging.info("Avvio dell'addestramento...")
    if console:
        console.put( "Avvio dell'addestramento...\n")
    
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        validation_data=(X_test, y_test),
        batch_size=64,
        callbacks=callbacks
    )
    
    val_accuracy = history.history['val_accuracy'][-1] * 100
    
    return model, timestamp, val_accuracy

# Funzione per salvare l'accuratezza su un file CSV
def save_accuracy_to_csv(timestamp, accuracy, file_path='model_accuracies.csv', console=None):
    header = ['Timestamp', 'Accuracy']
    data = [[timestamp, accuracy]]
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(data)
    message = f"Accuratezza salvata nel file {file_path}"
    logging.info(message)
    if console:
        console.put( message + '\n')

# Funzione per salvare il modello e il tokenizer
def save_model_and_tokenizer(model, tokenizer, max_len, timestamp, console=None):
    final_model_path = f'models/sentiment_analysis_{timestamp}_preaddestrato.h5'
    model.save(final_model_path)
    logging.info(f"Modello salvato correttamente in {final_model_path}.")
    if console:
        console.put( f"Modello salvato correttamente in {final_model_path}.\n")
    
    tokenizer_path = 'models/tokenizer.pkl'
    save_pickle(tokenizer, tokenizer_path)
    logging.info(f"Tokenizer salvato correttamente in {tokenizer_path}.")
    if console:
        console.put( f"Tokenizer salvato correttamente in {tokenizer_path}.\n")
    
    padding_path = 'models/padding.pkl'
    save_pickle(max_len, padding_path)
    logging.info(f"Padding salvato correttamente in {padding_path}.")
    if console:
        console.put( f"Padding salvato correttamente in {padding_path}.\n")

# Funzione principale per addestrare il modello di analisi del sentiment
def train_sentiment_analysis_model(file_path, embedding_file_path, console=None, embedding_dim=300):
    logging.info("Avvio dell'addestramento del modello di analisi del sentiment...")
    if console:
        console.put( "Avvio dell'addestramento del modello di analisi del sentiment...\n")
    configure_gpu(console)

    clean_data_file = file_path.replace('.csv', '_clean.pkl')
    tokenized_data_file = file_path.replace('.csv', '_tokenized.pkl')
    padded_data_file = file_path.replace('.csv', '_padded.pkl')

    df_train = load_preprocessed_data(clean_data_file)
    if df_train is None:
        df_train = load_and_preprocess_data(file_path, console)
        if df_train is None:
            logging.error("Errore nel caricamento o nella pulizia dei dati. Processo terminato.")
            if console:
                console.put( "Errore nel caricamento o nella pulizia dei dati. Processo terminato.\n")
            return None, None
        save_preprocessed_data(df_train, clean_data_file)

    token_data = load_preprocessed_data(tokenized_data_file)
    if token_data is None:
        optimal_num_words = calculate_optimal_num_words(df_train)
        tokenizer, clean_sequences, clean_labels = tokenize_reviews(df_train, optimal_num_words, console)
        token_data = (tokenizer, clean_sequences, clean_labels)
        save_preprocessed_data(token_data, tokenized_data_file)
    else:
        tokenizer, clean_sequences, clean_labels = token_data

    padded_data = load_preprocessed_data(padded_data_file)
    if padded_data is None:
        X_train, X_val, X_test, y_train, y_val, y_test, max_len = pad_sequences_and_split_data(clean_sequences, clean_labels, console=console)
        padded_data = (X_train, X_val, X_test, y_train, y_val, y_test, max_len)
        save_preprocessed_data(padded_data, padded_data_file)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test, max_len = padded_data

    embedding_matrix = load_pretrained_embeddings(embedding_file_path, tokenizer.word_index, embedding_dim)

    model = build_model(embedding_matrix, max_len, embedding_dim, console)

    model, timestamp, _ = train_model(model, X_train, y_train, X_val, y_val, console)

    save_model_and_tokenizer(model, tokenizer, max_len, timestamp, console)

    logging.info("Valutazione del modello sui dati di test...")
    if console:
        console.put( "Valutazione del modello sui dati di test...\n")
    _, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    test_accuracy_percent = test_accuracy * 100
    logging.info(f"Accuratezza modello: {test_accuracy_percent:.2f}%")
    logging.info(f"Precisione modello: {test_precision:.2f}")
    logging.info(f"Recall modello: {test_recall:.2f}")
    if console:
        console.put( f"Accuratezza modello: {test_accuracy_percent:.2f}%\n")
        console.put( f"Precisione modello: {test_precision:.2f}\n")
        console.put( f"Recall modello: {test_recall:.2f}\n")

    save_accuracy_to_csv(timestamp, test_accuracy_percent, console=console)
    
    logging.info(f"Accuratezza sui dati di test: {test_accuracy_percent:.2f}%")
    if console:
        console.put( f"Accuratezza sui dati di test: {test_accuracy_percent:.2f}%\n")

    return timestamp, test_accuracy_percent

