import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Assicurati di aver scaricato le risorse necessarie di NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Funzione per espandere le contrazioni
def expand_contractions(text):
    contractions = {
        "n't": " not", "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'t": " not", "'ve": " have", "'m": " am",
        "aren't": "are not", "can't": "cannot", "could've": "could have", "couldn't": "could not",
        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
        "he's": "he is", "how's": "how is", "I'd": "I would", "I'll": "I will", "I'm": "I am",
        "I've": "I have", "isn't": "is not", "it'd": "it would", "it's": "it is", "let's": "let us",
        "mightn't": "might not", "mustn't": "must not", "needn't": "need not", "shan't": "shall not",
        "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have",
        "shouldn't": "should not", "that'll": "that will", "there's": "there is", "they'd": "they would",
        "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not",
        "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not",
        "what'll": "what will", "what's": "what is", "when's": "when is", "where's": "where is",
        "who's": "who is", "won't": "will not", "would've": "would have", "wouldn't": "would not",
        "y'all": "you all", "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have"
    }
    for contraction, expansion in contractions.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)
    return text

# Inizializzazione delle risorse
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if isinstance(text, str):
        # Espansione delle contrazioni
        text = expand_contractions(text)
        # Rimozione di caratteri non alfanumerici e parole brevi
        text = re.sub(r'\b\w{1,2}\b|[^\w\s]', '', text)
        # Rimozione spazi extra e conversione in minuscolo
        text = re.sub(r'\s+', ' ', text).strip().lower()
        # Lemmatizzazione e rimozione stop words
        words = (lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
        return ' '.join(words)
    return ""
