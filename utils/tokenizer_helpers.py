from keras.preprocessing.text import Tokenizer

def calculate_optimal_num_words(df_train):
    print("Calcolo del numero ottimale di parole...")
    reviews = df_train['reviewText'].tolist()

    # Tokenizzazione del testo
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews)

    # Calcolo della frequenza delle parole
    word_freq = tokenizer.word_counts
    # Ottiene un dizionario con la frequenza di ogni parola nel testo.
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Calcolo della frequenza cumulativa
    total_words = sum(word_freq.values())
    # Calcola il numero totale di parole nel dataset.
    cumulative_coverage = 0
    optimal_num_words = 0
    for i, (word, freq) in enumerate(sorted_word_freq):
        cumulative_coverage += freq
        if cumulative_coverage / total_words >= 0.95:  # Copertura del 95%
            optimal_num_words = i + 1
            break

    # Itera attraverso le parole ordinate calcolando la copertura cumulativa. 
    # Se la copertura cumulativa raggiunge o supera il 95%, salva l'indice corrente come numero ottimale di parole.


    print(f"Numero ottimale di parole calcolato: {optimal_num_words}")
    return optimal_num_words
