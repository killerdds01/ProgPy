# Moduli standard
import os
import io
import datetime

# Moduli Flask e estensioni
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash # Per la gestione sicura delle password

# Moduli PyTorch per il modello
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Moduli MongoDB
from pymongo import MongoClient
from bson.objectid import ObjectId # Necessario per convertire l'ID di MongoDB

# --- Inizializzazione dell'Applicazione Flask ---
app = Flask(__name__)

# Configurazione della chiave segreta dell'applicazione.
# Questa chiave è fondamentale per la sicurezza delle sessioni (cookie) e per la protezione CSRF.
# È imperativo che questa chiave sia una stringa lunga, casuale e mantenuta segreta.
# Per la distribuzione online e su repository pubblici (es. GitHub),
# la chiave DEVE essere letta da una variabile d'ambiente e MAI committata direttamente nel codice.
# Esempio per generare una chiave robusta: python -c "import os; print(os.urandom(24).hex())"
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'e00d87cee9282c6458684cec5045dac65768c6db28968f2d!')

# Abilita Cross-Origin Resource Sharing (CORS) per l'applicazione.
# Permette al frontend (servito da Flask) di effettuare richieste all'API Flask stessa.
CORS(app) 

# --- Inizializzazione Flask-Login ---
# Configura l'estensione Flask-Login per gestire le sessioni utente.
login_manager = LoginManager()
login_manager.init_app(app)
# Specifica la vista di login a cui reindirizzare gli utenti non autenticati
login_manager.login_view = 'login' 

# --- Connessione al Database MongoDB ---
# Stabilisce una connessione al server MongoDB locale.
# In caso di errore di connessione, l'applicazione termina per prevenire malfunzionamenti.
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client.smart_waste # Seleziona il database 'smart_waste'
    predictions_collection = db.predictions # Collezione per le previsioni di classificazione dei rifiuti
    users_collection = db.users # Collezione per la gestione degli utenti registrati
    print("Connessione a MongoDB riuscita.")
except Exception as e:
    print(f"Errore di connessione a MongoDB: {e}")
    exit()

# --- Definizione del Modello Utente per Flask-Login ---
# Questa classe implementa i requisiti di Flask-Login per la gestione degli utenti.
class User(UserMixin):
    def __init__(self, user_data):
        # L'ID dell'utente è l'ObjectId di MongoDB convertito in stringa.
        self.id = str(user_data['_id']) 
        self.username = user_data['username']
        # L'hash della password viene memorizzato per la verifica, non la password in chiaro.
        self.password_hash = user_data['password'] 

    def get_id(self):
        # Metodo richiesto da Flask-Login per ottenere l'ID univoco dell'utente.
        return self.id

@login_manager.user_loader
def load_user(user_id):
    # Callback utilizzata da Flask-Login per ricaricare un utente dalla sessione.
    # Cerca l'utente nel database MongoDB tramite il suo ID.
    user_data = users_collection.find_one({"_id": ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

# --- Configurazione e Caricamento del Modello di Machine Learning ---
# Definisce le classi di rifiuto supportate dal modello.
CLASS_NAMES = ['plastica', 'carta', 'vetro', 'organico', 'indifferenziato']
NUM_CLASSES = len(CLASS_NAMES)

# Specifica il dispositivo su cui il modello verrà eseguito (CPU per compatibilità generale).
device = torch.device("cpu") 

# Percorso del file del modello PyTorch pre-addestrato.
MODEL_PATH = 'best_model_finetuned_light.pth' 

# Inizializza l'architettura del modello (ResNet18) senza pesi pre-addestrati iniziali.
model = models.resnet18(weights=None)

# Modifica l'ultimo strato (fully connected) del modello per adattarlo al numero di classi del dataset.
num_ftrs = model.fc.in_features 
model.fc = nn.Linear(num_ftrs, NUM_CLASSES) 

# Carica i pesi del modello dal file .pth.
# Gestisce errori se il file non viene trovato o se ci sono problemi durante il caricamento.
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device) # Sposta il modello sul dispositivo specificato.
    model.eval() # Imposta il modello in modalità di valutazione (disabilita dropout/batchnorm per inferenza).
    print(f"Modello caricato correttamente da {MODEL_PATH}, su dispositivo {device}.")
except FileNotFoundError:
    print(f"Errore: il file {MODEL_PATH} non è stato trovato. Assicurati che il percorso sia corretto.")
    exit() 
except Exception as e:
    print(f"Errore durante il caricamento del modello: {str(e)}")
    exit() 

# --- Configurazione delle Trasformazioni per le Immagini ---
# Queste trasformazioni pre-processano le immagini in ingresso per renderle compatibili con il modello.
transform = transforms.Compose([
    transforms.Resize((256)),  # Ridimensiona il lato più corto dell'immagine a 256 pixel.
    transforms.CenterCrop(224),  # Ritaglia l'immagine al centro per ottenere un quadrato di 224x224.
    transforms.ToTensor(),       # Converte l'immagine PIL in un tensore PyTorch e normalizza i pixel a [0,1].
    # Normalizza il tensore con media e deviazione standard predefinite per i modelli ImageNet.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Rotte dell'Applicazione ---

@app.route('/')
def home():
    # Renderizza la pagina principale dell'applicazione, dedicata alla classificazione.
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    # Gestisce la registrazione di nuovi utenti.
    # Se l'utente è già autenticato, viene reindirizzato alla home.
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Verifica se il nome utente esiste già nel database.
        if users_collection.find_one({'username': username}):
            flash('Nome utente già esistente. Scegli un altro nome.', 'error')
            return render_template('register.html')

        # Hashing della password prima di salvarla per garantire la sicurezza.
        hashed_password = generate_password_hash(password)
        
        # Inserisce il nuovo utente (username e password hashata) nel database.
        users_collection.insert_one({'username': username, 'password': hashed_password})
        flash('Registrazione avvenuta con successo! Ora puoi effettuare il login.', 'success')
        return redirect(url_for('login'))
    # Mostra il form di registrazione per le richieste GET.
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Gestisce l'accesso degli utenti esistenti.
    # Se l'utente è già autenticato, viene reindirizzato alla home.
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Cerca l'utente nel database MongoDB.
        user_data = users_collection.find_one({'username': username})

        # Verifica la password fornita con l'hash salvato nel database.
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user) # Effettua il login dell'utente tramite Flask-Login.
            flash('Login effettuato con successo!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Nome utente o password non validi.', 'error')
    # Mostra il form di login per le richieste GET.
    return render_template('login.html')

@app.route('/logout')
@login_required # Questa rotta richiede che l'utente sia autenticato per poter effettuare il logout.
def logout():
    # Effettua il logout dell'utente dalla sessione.
    logout_user() 
    flash('Logout effettuato con successo.', 'info')
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required # Questa rotta richiede che l'utente sia autenticato per poter classificare un'immagine.
def predict():
    # Verifica se la richiesta contiene un file immagine.
    if 'file' not in request.files:
        print("Nessun file immagine inviato.")
        return jsonify({'error': 'Nessun file immagine inviato.'}), 400

    file = request.files['file']
    # Verifica se il nome del file è vuoto (nessun file selezionato).
    if file.filename == '':
        print("Nessun file selezionato.")
        return jsonify({'error': 'Nessun file selezionato.'}), 400

    if file:
        try: 
            # Legge i byte dell'immagine dal file caricato.
            img_bytes = file.read()
            # Apre l'immagine con Pillow e la converte in formato RGB per garantire 3 canali.
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # Applica le trasformazioni predefinite all'immagine.
            input_tensor = transform(img)
            # Aggiunge una dimensione batch al tensore (necessaria per il modello).
            input_batch = input_tensor.unsqueeze(0)  
            input_batch = input_batch.to(device) # Sposta il tensore sul dispositivo specificato (CPU).

            with torch.no_grad():  # Disabilita il calcolo dei gradienti per ottimizzare l'inferenza.
                output = model(input_batch)  # Esegue la previsione del modello.
                # Calcola le probabilità delle classi usando la funzione softmax.
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                # Ottiene la probabilità più alta e l'indice della classe predetta.
                predicted_prob, predicted_idx = torch.max(probabilities, 0)
                # Ottiene il nome della classe predetta dal mapping CLASS_NAMES.
                predicted_class = CLASS_NAMES[predicted_idx.item()]  
                
            # Crea un documento con i dati della previsione da salvare nel database MongoDB.
            prediction_data = { 
                'class': predicted_class, # La classe di rifiuto predetta.
                'confidence': f"{predicted_prob.item() * 100:.2f}%", # La confidenza della previsione in percentuale.
                'timestamp': datetime.datetime.now(), # Timestamp della previsione.
                'image_filename': file.filename, # Nome originale del file immagine caricato.
                'user_id': current_user.id # Associa la previsione all'ID dell'utente autenticato.
            }
            predictions_collection.insert_one(prediction_data) # Salva il documento nella collezione 'predictions'.

            # Ritorna la previsione come risposta JSON al client.
            return jsonify({
                'class': predicted_class, 
                'confidence': f"{predicted_prob.item() * 100:.2f}%" 
            })
        except Exception as e:
            # Cattura e stampa eventuali errori durante l'elaborazione dell'immagine o la previsione.
            print(f"Si è verificato un errore durante la previsione: {e}")
            # Ritorna un messaggio di errore JSON al browser con stato HTTP 500.
            return jsonify({'error': f'Errore durante la previsione: {str(e)}'}), 500
        
    # Gestisce il caso in cui il file non è stato processato per un motivo sconosciuto.
    return jsonify({'error': 'Errore sconosciuto durante la previsione.'}), 500

@app.route('/dashboard')
@login_required # Questa rotta richiede che l'utente sia autenticato per accedere alla dashboard.
def dashboard():
    # Recupera solo le previsioni associate all'utente attualmente autenticato.
    # Le previsioni sono ordinate dalla più recente alla meno recente.
    predictions = list(predictions_collection.find({'user_id': current_user.id}).sort('timestamp', -1))
    
    # Inizializza un contatore per ogni classe di rifiuto.
    class_counts = {class_name: 0 for class_name in CLASS_NAMES}
    # Popola il contatore iterando sulle previsioni recuperate.
    for p in predictions:
        # Assicurati che la chiave 'class' esista nel documento di predizione prima di accedere.
        if 'class' in p:
            class_counts[p['class']] += 1
    # Renderizza la pagina della dashboard, passando le previsioni e i conteggi delle classi al template.
    return render_template('dashboard.html', predictions=predictions, class_counts=class_counts)

# --- Avvio dell'Applicazione Flask ---
if __name__ == '__main__':
    # Avvia l'applicazione Flask.
    # debug=True: Abilita il reloader automatico (riavvia il server a ogni modifica del codice) e il debugger.
    # host='127.0.0.1': Rende l'app accessibile solo localmente 
    # port=5000: Specifica la porta su cui il server sarà in ascolto.
    app.run(debug=True, host='127.0.0.1', port=5000)