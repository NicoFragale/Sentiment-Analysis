import logging
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from train_model import train_sentiment_analysis_model
from apply_model import apply_sentiment_analysis_thread
from apply_model_rotten_tomatoes import apply_model_on_rotten_tomatoes_thread
import threading
import queue

# Configurazione del logging per tkinter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funzione per selezionare un file tramite GUI
def select_file_gui(title="Seleziona un file"):
    try:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title=title, filetypes=[("All files", "*.*")])
        root.destroy()
        return file_path
    except Exception as e:
        logging.error(f"Errore durante la selezione del file: {e}")
        messagebox.showerror("Errore", f"Errore durante la selezione del file: {e}")
        return None

# Funzione per selezionare una directory tramite GUI
def select_save_location_gui(title="Seleziona la cartella di destinazione"):
    try:
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(title=title)
        root.destroy()
        return folder_path
    except Exception as e:
        logging.error(f"Errore durante la selezione della cartella: {e}")
        messagebox.showerror("Errore", f"Errore durante la selezione della cartella: {e}")
        return None

# Funzione per aggiornare la console leggendo dalla console_queue
def update_console(console, console_queue):
    try:
        while True:
            message = console_queue.get_nowait()
            console.insert(tk.END, message + "\n")
            console.see(tk.END)
    except queue.Empty:
        pass
    root.after(100, update_console, console, console_queue)

# Funzione per addestrare il modello
def train_model_thread(file_path, embedding_file_path, console_queue):
    timestamp, accuracy = train_sentiment_analysis_model(file_path, embedding_file_path, console_queue)
    console_queue.put(f"Addestramento completato. Timestamp: {timestamp}, Accuratezza: {accuracy:.2f}%\n")
    
def start_training(console_queue):    
    file_path = select_file_gui("Seleziona il file CSV di addestramento")
    pretrained_embedding_path = select_file_gui("Seleziona il file contenente il modello preaddestrato")
    console_queue.put("Avvio dell'addestramento...\n")

    training_thread = threading.Thread(target=train_model_thread, args=(file_path, pretrained_embedding_path, console_queue))
    training_thread.start()

# Funzione per applicare il modello
def apply_model_thread(console_queue):
    def task():
        model_path = select_file_gui("Seleziona il modello di analisi del sentiment")
        tokenizer_path = select_file_gui("Seleziona il file contenente le informazioni per il tokenizer")
        padding_path =  select_file_gui("Seleziona il file contenente le informazioni per il padding")
        if model_path and tokenizer_path and padding_path:
            logging.info(f"Modello selezionato: {model_path}")
            console_queue.put("Modello selezionato: " + model_path)
            console_queue.put("File per il tokenizer selezionato: " + tokenizer_path)
            console_queue.put("File per il padding selezionato: " + padding_path)
            data_path = select_file_gui("Seleziona il file CSV di dati su cui applicare il modello")
            if data_path:
                logging.info(f"File di dati selezionato: {data_path}")
                console_queue.put("File di dati selezionato: " + data_path)
                try:
                    apply_sentiment_analysis_thread(model_path, data_path, console_queue, tokenizer_path, padding_path)
                    console_queue.put("Analisi del sentiment completata con successo!")
                except Exception as e:
                    logging.error(f"Errore durante l'applicazione del modello: {e}")
                    console_queue.put(f"Errore durante l'applicazione del modello: {e}")
            else:
                logging.info("Nessun file di dati selezionato.")
                console_queue.put("Nessun file di dati selezionato.")
        else:
            logging.info("Nessun modello selezionato.")
            console_queue.put("Nessun modello selezionato.")
    threading.Thread(target=task, daemon=True).start()

# Funzione per applicare il modello ai commenti di Rotten Tomatoes
def apply_model_to_rotten_tomatoes_thread(console_queue):
    def task():
        model_path = select_file_gui("Seleziona il modello di analisi del sentiment")
        tokenizer_path = select_file_gui("Seleziona il file contenente le informazioni per il tokenizer")
        padding_path =  select_file_gui("Seleziona il file contenente le informazioni per il padding")
        if model_path and tokenizer_path and padding_path:
            logging.info(f"Modello selezionato: {model_path}")
            console_queue.put("Modello selezionato: " + model_path)
            console_queue.put("File per il tokenizer selezionato: " + tokenizer_path)
            console_queue.put("File per il padding selezionato: " + padding_path)
            def submit_url(url):
                if url:
                    logging.info(f"URL selezionato: {url}")
                    try:
                        results = apply_model_on_rotten_tomatoes_thread(url, model_path, console_queue, tokenizer_path, padding_path)
                        console_queue.put("Analisi del sentiment completata con successo! Risultati: " + str(results))
                    except Exception as e:
                        logging.error(f"Errore durante l'applicazione del modello: {e}")
                        console_queue.put(f"Errore durante l'applicazione del modello: {e}")
                else:
                    console_queue.put("Nessun URL inserito.")
            def ask_for_url():
                url = tk.simpledialog.askstring("URL di Rotten Tomatoes", "Inserisci l'URL di Rotten Tomatoes:")
                if url:
                    print(f"URL inserito: {url}")
                    submit_url(url)
            root.after(0, ask_for_url)
        else:
            logging.info("Modello non selezionato.")
            console_queue.put("Modello non selezionato.")
    threading.Thread(target=task, daemon=True).start()

# Funzione per il menu principale dell'applicazione
def main_menu():
    global root
    root = tk.Tk()
    root.title("Analisi del Sentiment")

    # Creazione della console_queue
    console_queue = queue.Queue()

    def on_train_button_click():
        start_training( console_queue)

    def on_apply_button_click():
        apply_model_thread( console_queue)

    def on_apply_to_rotten_tomatoes_click():
        apply_model_to_rotten_tomatoes_thread( console_queue)

    def on_exit_button_click():
        root.destroy()

    tk.Label(root, text="Menu", font=("Helvetica", 16)).pack(pady=10)

    tk.Button(root, text="Allenare il modello di analisi del sentiment", command=on_train_button_click, width=53).pack(pady=5, padx=10)
    tk.Button(root, text="Scegli un modello di analisi del sentiment e applicalo a un set di dati", command=on_apply_button_click, width=53).pack(pady=5, padx=10)
    tk.Button(root, text="Analisi del sentiment dei commenti di Rotten Tomatoes", command=on_apply_to_rotten_tomatoes_click, width=53).pack(pady=5, padx=10)
    tk.Button(root, text="Uscire", command=on_exit_button_click).pack(pady=5, padx=10)

    console_label = tk.Label(root, text="Console Output:", font=("Helvetica", 12))
    console_label.pack(pady=10)

    global console
    console = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20, font=("Helvetica", 10))
    console.pack(pady=10)

    # Avvia l'aggiornamento della console
    root.after(100, update_console, console, console_queue)

    root.mainloop()

if __name__ == "__main__":
    main_menu()
