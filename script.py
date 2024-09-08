import pandas as pd

# Carica il file delle predizioni
file_path = 'predictions_test.csv'
df_predictions = pd.read_csv(file_path, encoding='iso-8859-1')

# Calcola l'accuratezza
threshold = 0.5  # Soglia per considerare una predizione positiva
df_predictions['predicted_sentiment'] = df_predictions['predictions'].apply(lambda x: 1 if x >= threshold else 0)

correct_predictions = (df_predictions['sentiment'] == df_predictions['predicted_sentiment'])
accuracy = correct_predictions.mean() * 100
print(f"Accuratezza del modello: {accuracy:.2f}%")

# Calcola l'errore medio
mean_error = (df_predictions['predictions'] - df_predictions['sentiment']).abs().mean()
print(f"Errore medio: {mean_error:.4f}")

# Calcola la percentuale di predizioni correttamente orientate
correct_orientation = (df_predictions['predicted_sentiment'] == df_predictions['sentiment']).mean() * 100
print(f"Percentuale di predizioni correttamente orientate: {correct_orientation:.2f}%")
