import pandas as pd
import logging
import tkinter as tk
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from utils.text_cleaning import clean_text
from utils.file_helpers import load_pickle
import threading
import queue

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_resources(model_path, tokenizer_path, padding_path, console_queue):
    try:
        tokenizer = load_pickle(tokenizer_path)
        logging.info("Tokenizer caricato.")
        console_queue.put("Tokenizer caricato.")
        
        max_len = load_pickle(padding_path)
        logging.info("Max_len caricato.")
        console_queue.put("Max_len caricato.")
        
        model = load_model(model_path)
        logging.info("Modello caricato.")
        console_queue.put("Modello caricato.")
        
        return tokenizer, max_len, model
    except Exception as e:
        logging.error(f"Errore nel caricamento delle risorse: {e}")
        console_queue.put(f"Errore nel caricamento delle risorse: {e}")
        raise

def preprocess_data(data_path, console_queue):
    try:
        df_test = pd.read_csv(data_path, encoding='iso-8859-1')
        logging.info("Dati di test caricati.")
        console_queue.put("Dati di test caricati.")
        
        df_test = df_test[df_test['sentiment'] != 'neutral']
        logging.info("Record con sentiment 'neutral' rimossi.")
        console_queue.put("Record con sentiment 'neutral' rimossi.")
        
        sentiment_mapping = {'positive': 1, 'negative': 0}
        df_test['sentiment'] = df_test['sentiment'].map(sentiment_mapping)
        logging.info("Sentiment mappati in valori numerici.")
        console_queue.put("Sentiment mappati in valori numerici.")
        
        df_test['review'] = df_test['review'].apply(clean_text)
        logging.info("Testo delle recensioni pulito.")
        console_queue.put("Testo delle recensioni pulito.")
        
        df_test['is_empty'] = df_test['review'].apply(lambda x: 1 if len(x.strip()) == 0 else 0)
        df_test = df_test[df_test['is_empty'] == 0].drop(columns=['is_empty'])
        logging.info("Recensioni vuote rimosse.")
        console_queue.put("Recensioni vuote rimosse.")
        
        return df_test
    except Exception as e:
        logging.error(f"Errore nella pre-elaborazione dei dati: {e}")
        console_queue.put(f"Errore nella pre-elaborazione dei dati: {e}")
        raise

def tokenize_and_pad_sequences(tokenizer, texts, max_len, console_queue):
    sequences = tokenizer.texts_to_sequences(texts)
    logging.info("Recensioni tokenizzate.")
    console_queue.put("Recensioni tokenizzate.")
    
    sequences = [seq for seq in sequences if len(seq) > 0]
    logging.info(f"Numero di sequenze non vuote: {len(sequences)}")
    console_queue.put(f"Numero di sequenze non vuote: {len(sequences)}")
    
    data_padded = pad_sequences(sequences, maxlen=max_len)
    logging.info("Sequenze delle recensioni padded.")
    console_queue.put("Sequenze delle recensioni padded.")
    
    return data_padded

def predict_sentiments(model, data_padded, console_queue):
    predictions = model.predict(data_padded)
    logging.info("Predizione del sentiment completata.")
    console_queue.put("Predizione del sentiment completata.")
    return predictions.flatten()

def calculate_statistics(predictions, df_predictions, console_queue):
    try:
        # Verifica che la colonna 'sentiment' esista nel DataFrame
        if 'sentiment' not in df_predictions.columns:
            raise ValueError("La colonna 'sentiment' non è presente nel DataFrame.")
        
        # Definisci la soglia
        threshold = 0.5
        
        # Converti le previsioni in una pandas Series
        predictions_series = pd.Series(predictions)
        
        # Verifica che la lunghezza delle previsioni corrisponda al numero di righe nel DataFrame
        if len(predictions_series) != len(df_predictions):
            raise ValueError("La lunghezza delle previsioni non corrisponde al numero di righe nel DataFrame.")
        
        # Converti le previsioni in 0 o 1 basate sulla soglia
        predictions_binary = predictions_series.apply(lambda x: 1 if x >= threshold else 0)
        
        # Converti le etichette nel DataFrame in 0 o 1 basate sulla soglia
        df_predictions['sentiment_binary'] = df_predictions['sentiment']
        
        # Calcola la correttezza delle previsioni
        correct_predictions = df_predictions['sentiment_binary'] == predictions_binary
        accuracy = correct_predictions.mean() * 100
        console_queue.put(f"Accuratezza del modello: {accuracy:.2f}%")
        
        # Calcola l'errore medio
        mean_error = (df_predictions['sentiment_binary'] - predictions_binary).abs().mean()
        console_queue.put(f"Errore medio: {mean_error:.4f}")
        
        # Calcola la percentuale di orientamenti corretti
        correct_orientation = correct_predictions.mean() * 100
        console_queue.put(f"Percentuale di predizioni correttamente orientate: {correct_orientation:.2f}%")
        
    except Exception as e:
        logging.error(f"Errore nel calcolo delle statistiche: {e}")
        console_queue.put(f"Errore nel calcolo delle statistiche: {e}")


def apply_sentiment_analysis(model_path, data_path, console_queue, tokenizer_path='models/tokenizer.pkl', padding_path='models/padding.pkl', output_file='predictions_test.csv'):
    try:
        tokenizer, max_len, model = load_resources(model_path, tokenizer_path, padding_path, console_queue)
        df_test = preprocess_data(data_path, console_queue)
        
        data_padded = tokenize_and_pad_sequences(tokenizer, df_test['review'], max_len, console_queue)
        predictions = predict_sentiments(model, data_padded, console_queue)
        
        df_test = df_test.head(len(predictions))
        
        # Passa il DataFrame invece del percorso del file
        calculate_statistics(predictions, df_test, console_queue)
        
    except Exception as e:
        logging.error(f"Errore durante l'applicazione dell'analisi del sentiment: {e}")
        console_queue.put(f"Errore durante l'applicazione dell'analisi del sentiment: {e}")


def apply_sentiment_analysis_thread(model_path, data_path, console_queue, tokenizer_path, padding_path):
    def task():
        apply_sentiment_analysis(model_path, data_path, console_queue, tokenizer_path, padding_path)
    
    threading.Thread(target=task, daemon=True).start()
