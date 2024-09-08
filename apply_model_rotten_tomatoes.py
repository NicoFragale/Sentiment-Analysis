import pickle
import requests
import threading
import logging
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from utils.text_cleaning import clean_text
from bs4 import BeautifulSoup

# Configurazione base del logging
logging.basicConfig(
    level=logging.DEBUG,  # Imposta il livello di logging su DEBUG per ottenere messaggi dettagliati
    format='%(asctime)s - %(levelname)s - %(message)s',  # Definisce il formato del messaggio di log
    handlers=[
        logging.StreamHandler()  # Stampa i log sulla console
    ]
)

def load_model_and_tokenizer(model_path, console_queue, tokenizer_path, padding_path):
    """
    Carica il modello di sentiment analysis e il tokenizer utilizzato per preprocessare i testi.
    """
    console_queue.put("Inizio il caricamento del modello e del tokenizer.")
    
    # Carica il modello
    try:
        model = load_model(model_path)
        console_queue.put("Modello caricato con successo.")
    except Exception as e:
        console_queue.put(f"Errore durante il caricamento del modello: {e}")
        raise
    
    # Carica il tokenizer
    try:
        with open(tokenizer_path, 'rb') as file:
            tokenizer = pickle.load(file)
        console_queue.put("Tokenizer caricato con successo.")
    except Exception as e:
        console_queue.put(f"Errore durante il caricamento del tokenizer: {e}")
        raise
    
    # Carica il padding (lunghezza massima delle sequenze)
    try:
        with open(padding_path, 'rb') as file:
            max_len = pickle.load(file)
        console_queue.put("Padding caricato con successo.")
    except Exception as e:
        console_queue.put(f"Errore durante il caricamento del padding: {e}")
        raise
    
    return model, tokenizer, max_len

def extract_movie_id(page_url, console_queue):
    """
    Estrae l'ID del film dalla pagina del URL fornito.
    """
    try:
        response = requests.get(page_url)
        response.raise_for_status()  # Verifica che la richiesta HTTP sia andata a buon fine

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Trova lo script che contiene l'ID del film
        script_tag = soup.find('script', string=lambda text: text and 'titleId' in text)
        if script_tag:
            script_content = script_tag.string
            title_id_start = script_content.find('titleId":"') + len('titleId":"')
            title_id_end = script_content.find('"', title_id_start)
            title_id = script_content[title_id_start:title_id_end]
        else:
            title_id = None
        
        return title_id
    except requests.exceptions.RequestException as e:
        console_queue.put(f"Errore durante l'estrazione dell'ID del film: {e}")
        return None
    except Exception as e:
        console_queue.put(f"Errore durante l'estrazione dell'ID del film: {e}")
        return None

def scrape_comments(page_url, console_queue):
    """
    Estrae i commenti delle recensioni dal sito utilizzando l'ID del film.
    """
    api_url = "https://www.rottentomatoes.com/napi/movie"
    movie_id = extract_movie_id(page_url, console_queue)
    
    if not movie_id:
        console_queue.put("Impossibile estrarre l'ID del film.")
        return []

    comments = []
    after = ''  # Il valore di 'after' per la paginazione

    while True:
        try:
            full_url = f"{api_url}/{movie_id}/reviews/all?after={after}&pageCount=20"
            response = requests.get(full_url)
            response.raise_for_status()  # Verifica che la richiesta HTTP sia andata a buon fine
            
            data = response.json()
            reviews = data.get('reviews', [])
            
            # Estrai il testo delle recensioni
            for review in reviews:
                review_text = review.get('quote', '').strip()
                if review_text:
                    comments.append(review_text)
            
            # Ottieni il token per la pagina successiva, se disponibile
            after = data.get('pageInfo', {}).get('endCursor')
            if not after:
                break
            
            console_queue.put(f"Recuperati {len(comments)} commenti finora.")
        
        except requests.exceptions.RequestException as e:
            console_queue.put(f"Errore durante il fetching della pagina: {e}")
            break
        except Exception as e:
            console_queue.put(f"Errore durante lo scraping: {e}")
            break
    
    console_queue.put(f"Totale commenti recuperati: {len(comments)}")
    return comments

def preprocess_comments(comments, tokenizer, max_len, console_queue):
    """
    Preprocessa i commenti pulendo il testo e convertendoli in sequenze di numeri.
    """
    console_queue.put("Inizio il preprocessing dei commenti.")
    try:
        # Pulisci i commenti
        clean_comments = [clean_text(comment) for comment in comments]
        # Converti i commenti puliti in sequenze di numeri
        sequences = tokenizer.texts_to_sequences(clean_comments)
        # Pad (riempi) le sequenze per avere una lunghezza uniforme
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        console_queue.put(f"Commenti preprocessati: {len(clean_comments)}, Sequenze: {padded_sequences.shape}")
        return padded_sequences
    except Exception as e:
        console_queue.put(f"Errore durante il preprocessing dei commenti: {e}")
        raise

def predict_sentiments(model, padded_sequences, console_queue):
    """
    Usa il modello di sentiment analysis per prevedere il sentiment dei commenti.
    """
    console_queue.put("Inizio la previsione dei sentimenti.")
    try:
        # Prevedi il sentiment dei commenti
        predictions = model.predict(padded_sequences)
        console_queue.put("Predizioni ottenute")
        # Converti le predizioni in etichette 'POSITIVE' o 'NEGATIVE'
        return ['POSITIVE' if pred > 0.5 else 'NEGATIVE' for pred in predictions]
    except Exception as e:
        console_queue.put(f"Errore durante la previsione dei sentimenti: {e}")
        raise

def display_results(comments, sentiments, console_queue):
    """
    Calcola e visualizza le statistiche sui commenti.
    """
    console_queue.put("Inizio la visualizzazione dei risultati.")

    # Calcola le statistiche
    num_comments = len(comments)
    num_positive = sentiments.count('POSITIVE')
    num_negative = sentiments.count('NEGATIVE')
    
    # Visualizza le statistiche
    console_queue.put(f"Numero totale di commenti: {num_comments}")
    console_queue.put(f"Numero di commenti positivi: {num_positive}")
    console_queue.put(f"Numero di commenti negativi: {num_negative}")
    
    return num_comments, num_positive, num_negative

def apply_model_on_rotten_tomatoes(url, model_path, console_queue, tokenizer_path, padding_path):
    """
    Funzione principale che coordina il flusso di lavoro:
    1. Estrae i commenti.
    2. Carica il modello e il tokenizer.
    3. Preprocessa i commenti.
    4. Prevede i sentimenti.
    5. Visualizza i risultati.
    """
    comments = scrape_comments(url, console_queue)
    model, tokenizer, max_len = load_model_and_tokenizer(model_path, console_queue, tokenizer_path, padding_path)
    padded_sequences = preprocess_comments(comments, tokenizer, max_len, console_queue)
    sentiments = predict_sentiments(model, padded_sequences, console_queue)
    numero_commenti, positivi, negativi = display_results(comments, sentiments, console_queue)
    result = f"Numero commenti : {numero_commenti} \nCommenti positivi : {positivi} \nCommenti negativi : {negativi} \nPercentuale positivi : {positivi/numero_commenti * 100} "
    return result

def apply_model_on_rotten_tomatoes_thread(url, model_path, console_queue, tokenizer_path, padding_path):
    """
    Avvia l'analisi del modello su Rotten Tomatoes in un thread separato.
    """
    def task():
        try:
            result = apply_model_on_rotten_tomatoes(url, model_path, console_queue, tokenizer_path, padding_path)
            console_queue.put(f"Analisi completata:\n{result}")
        except Exception as e:
            console_queue.put(f"Errore durante l'applicazione del modello: {e}")
    
    threading.Thread(target=task, daemon=True).start()
