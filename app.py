# Importa le librerie necessarie
import os
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Inizializza l'app Flask
app = Flask(__name__)

# --- Configura il modello pre-addestrato ---
CLASS_NAMES = ['plastica', 'carta', 'vetro', 'organico', 'indifferenziato']
NUM_CLASSES= len(CLASS_NAMES) # Il numero di classi deve essere 5

device = torch.device("cpu")

MODEL_PATH = 'best_model_finetuned_ligth.pth'

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

