import pickle
import os

# Funzione per salvare i dati preprocessati
def save_preprocessed_data(data, file_name):
    """
    Salva i dati preprocessati in un file pickle.

    Args:
        data: I dati da salvare.
        file_name: Il nome del file in cui salvare i dati.
    """
    # Crea la cartella 'preprocessed_data' se non esiste
    os.makedirs('preprocessed_data', exist_ok=True)
    
    # Crea il percorso completo del file all'interno della cartella 'preprocessed_data'
    full_path = os.path.join('preprocessed_data', file_name)
    
    # Apre il file in modalità scrittura binaria e salva i dati utilizzando pickle
    with open(full_path, 'wb') as file:
        pickle.dump(data, file)
    
    # Stampa un messaggio di conferma
    print(f"Dati preprocessati salvati in {file_name}")

# Funzione per caricare i dati preprocessati
def load_preprocessed_data(file_name):
    """
    Carica i dati preprocessati da un file pickle.

    Args:
        file_name: Il nome del file da cui caricare i dati.

    Returns:
        I dati caricati, o None se il file non esiste.
    """
    # Crea il percorso completo del file all'interno della cartella 'preprocessed_data'
    full_path = os.path.join('preprocessed_data', file_name)
    
    # Controlla se il file esiste
    if os.path.exists(full_path):
        # Apre il file in modalità lettura binaria e carica i dati utilizzando pickle
        with open(full_path, 'rb') as file:
            data = pickle.load(file)
        
        # Stampa un messaggio di conferma
        print(f"Dati preprocessati caricati da {file_name}")
        return data
    else:
        # Stampa un messaggio di errore se il file non esiste
        print(f"{file_name} non trovato.")
        return None
