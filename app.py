# Importa le librerie necessarie
import os
import io
import datetime # Importa la libreria datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Importa la libreria MongoDB
from pymongo import MongoClient
# Connettiti a MongoDB (di default è su localhost:27017)
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client.smart_waste
    predictions_collection = db.predictions
    print("Connessione a MongoDB riuscita.")
except Exception as e:
    print(f"Errore di connessione a MongoDB: {e}")
    exit()
    
# Inizializza l'app Flask
app = Flask(__name__)

CORS(app)  # Abilita CORS per consentire richieste da altri domini

# --- Configura il modello pre-addestrato ---
CLASS_NAMES = ['plastica', 'carta', 'vetro', 'organico', 'indifferenziato']
NUM_CLASSES= len(CLASS_NAMES) # Il numero di classi deve essere 5

device = torch.device("cpu")

MODEL_PATH = 'best_model_finetuned_light.pth'  # Percorso del file del modello addestrato

# Inizializza l'architettura del modello (ResNet18).
# Carichiamo un modello ResNet18 senza pesi pre-addestrati inizialmente, perché caricheremo i nostri pesi personalizzati.

model = models.resnet18(weights=None)

# Modifica l'ultimo strato per adattarlo al numero di classi del nostro dataset
num_ftrs = model.fc.in_features # Ottieni il numero di caratteristiche in ingresso dell'ultimo strato
model.fc = nn.Linear(num_ftrs, NUM_CLASSES) # Carica i pesi del modello addestrato

# Carica i pesi del modello dal file .pth
# map_locatio= device assicura che i pesi vengano caricati sul dispositivo corretto
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device) # Sposta il modello sul dispositivo specificato
    model.eval() # Imposta il modello in modalità di valutazione
    print(f"Modello caricato correttamente da {MODEL_PATH}, su dispositivo {device}.")
except FileNotFoundError:
    print(f"Errore: il file {MODEL_PATH} non è stato trovato. Assicurati che il percorso sia corretto.")
    exit() # Esci se il modello non può essere caricato
except Exception as e:
    print(f"Errore durante il caricamento del modello: {str(e)}")
    exit() # Esci se c'è un errore durante il caricamento del modello

# --- Configura le trasformazioni per le immagini ---
transform = transforms.Compose([
    transforms.Resize((256)),  # Ridimensiona l'immagine al lato piu corto
    transforms.CenterCrop(224),  # Ritaglia l'immagine al centro per ottenere un quadrato di 224x224
    transforms.ToTensor(),       # Converte l'immagine in un tensore
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza l'immagine
])

# --- Endpoint API ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Controlla se la richiesta contiene un file
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file immagine inviato.'}), 400

    file = request.files['file']
    # Controlla se il nome non è vuoto
    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato.'}), 400

    if file:
        try: 
            # Leggi i byte dell'immagine
            img_bytes = file.read()
            # Apri l'immagine con Pillow e converte in RGB
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # Applica le trasformazioni all'immagine
            input_tensor = transform(img)
            input_batch = input_tensor.unsqueeze(0)  # Aggiungi una dimensione batch

            # Sposta il batch sul dispositivo corretto
            input_batch = input_batch.to(device)

            with torch.no_grad():  # Disabilita il calcolo dei gradienti
                output = model(input_batch)  # Esegui la previsione
                # Ottieni le probabilità (softmax) e l'indice della classe predetta
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_prob, predicted_idx = torch.max(probabilities, 0)
                predicted_class = CLASS_NAMES[predicted_idx.item()]  # Ottieni il nome della classe predetta

                prediction_data = { 
                    'class': predicted_class, # Salva la classe predetta
                    'confidence': f"{predicted_prob.item() * 100:.2f}%", # Converti la probabilità in percentuale
                    'timestamp': datetime.datetime.now(), # Aggiungi un timestamp della previsione
                    'image_filename': file.filename # Salva il nome del file dell'immagine
                }
                predictions_collection.insert_one(prediction_data) # Salva la previsione nel database MongoDB

                return jsonify({
                    'class': predicted_class, 
                    'confidence': f"{predicted_prob.item() * 100:.2f}%" 
                })
        except Exception as e:
            # Stampa l'errore completo nel terminale per il debug
            print(f"Si è verificato un errore durante la previsione: {e}")
            # Ritorna un messaggio di errore JSON al browser
            return jsonify({'error': f'Errore durante la previsione: {str(e)}'}), 500
        
    return jsonify({'error': 'Errore sconosciuto durante la previsione.'}), 500

# --- Avvia l'app Flask ---
if __name__ == '__main__':

    app.run(debug=True, host='127.0.0.1', port=5000)  # Avvia l'applicazione Flask sulla porta 5000