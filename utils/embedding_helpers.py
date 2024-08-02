import numpy as np

def load_pretrained_embeddings(embedding_file_path, word_index, embedding_dim):
    print("Caricamento degli embeddings preaddestrati...")
    embeddings_index = {}

    with open(embedding_file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]

            # coefficienti (vettori di embedding) associati alla parola
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Crea una matrice di zeri con una dimensione pari al numero di parole nel dizionario più uno (per l'indice 0) e la dimensione degli embeddings
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)

        # Se esiste un vettore di embedding per la parola corrente, lo inserisce nella matrice degli embeddings
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("Embeddings preaddestrati caricati.")

    # Ritorna la matrice degli embeddings
    return embedding_matrix


'''
è una funzione che carica gli embeddings preaddestrati da un file 
e crea una matrice di embeddings che può essere 
utilizzata per inizializzare un layer di embedding in 
un modello di deep learning, migliorando così le prestazioni 
del modello rispetto all'inizializzazione casuale.
'''
