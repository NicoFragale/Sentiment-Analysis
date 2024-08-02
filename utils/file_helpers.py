import pickle

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

'''
per salvare e caricare oggetti complessi
come modelli di machine learning, dizionari, liste, 
o qualsiasi altro tipo di oggetto Python che può essere serializzato.
'''