# Moduli standard
import os
import io
import datetime
import uuid

# Moduli Flask e estensioni
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Moduli PyTorch per il modello
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Moduli MongoDB
from pymongo import MongoClient
from bson.objectid import ObjectId

# Carica le variabili d'ambiente dal file .env.
load_dotenv()

# --- Inizializzazione dell'Applicazione Flask ---
app = Flask(__name__)

# Configurazione della chiave segreta dell'applicazione.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'CHIAVE_DI_DEFAULT_NON_SICURA_IN_PRODUZIONE_CAMBIALA_SUBITO!')

# Imposta la cartella in cui verranno salvate le immagini caricate dagli utenti.
UPLOAD_FOLDER = 'uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Crea la cartella se non esiste.
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Creata la cartella di upload: {UPLOAD_FOLDER}")

CORS(app) 

# --- Inizializzazione Flask-Login ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 

# --- Connessione al Database MongoDB ---
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client.smart_waste
    predictions_collection = db.predictions
    users_collection = db.users
    print("Connessione a MongoDB riuscita.")
except Exception as e:
    print(f"Errore di connessione a MongoDB: {e}")
    exit()

# --- Definizione del Modello Utente per Flask-Login ---
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id']) 
        self.username = user_data['username']
        self.password_hash = user_data['password'] 

    def get_id(self):
        return self.id

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({"_id": ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

# --- Configurazione e Caricamento del Modello di Machine Learning ---
CLASS_NAMES = ['plastica', 'carta', 'vetro', 'organico', 'indifferenziato']
NUM_CLASSES = len(CLASS_NAMES)
device = torch.device("cpu") 
MODEL_PATH = 'best_model_finetuned_light.pth' 

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features 
model.fc = nn.Linear(num_ftrs, NUM_CLASSES) 

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Modello caricato correttamente da {MODEL_PATH}, su dispositivo {device}.")
except FileNotFoundError:
    print(f"Errore: il file {MODEL_PATH} non è stato trovato. Assicurati che il percorso sia corretto.")
    exit() 
except Exception as e:
    print(f"Errore durante il caricamento del modello: {str(e)}")
    exit() 

# --- Configurazione delle Trasformazioni per le Immagini ---
transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Rotte dell'Applicazione ---

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if users_collection.find_one({'username': username}):
            flash('Nome utente già esistente. Scegli un altro nome.', 'error')
            return render_template('register.html')

        hashed_password = generate_password_hash(password)
        
        users_collection.insert_one({'username': username, 'password': hashed_password})
        flash('Registrazione avvenuta con successo! Ora puoi effettuare il login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_data = users_collection.find_one({'username': username})

        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)
            flash('Login effettuato con successo!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Nome utente o password non validi.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('_flashes', None) 
    logout_user() 
    flash('Logout effettuato con successo.', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """
    Gestisce l'upload di un'immagine, la salva, esegue una predizione
    del modello di classificazione e salva i risultati nel database.
    """
    # 1. Verifica che un file sia stato inviato
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file immagine inviato.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato.'}), 400

    if file:
        try:
            # 2. Genera un nome di file univoco e crea il percorso di salvataggio
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            # Usa os.path.join per costruire un percorso compatibile con tutti i sistemi operativi
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # 3. SALVA IL FILE SUL DISCO PER PRIMO
            # Questo è il passo cruciale per garantire che il file non sia vuoto
            file.save(filepath)
            
            # 4. Apri l'immagine SALVATA per l'elaborazione del modello
            img = Image.open(filepath).convert('RGB')
            input_tensor = transform(img)
            input_batch = input_tensor.unsqueeze(0)
            input_batch = input_batch.to(device)

            with torch.no_grad():
                output = model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_prob, predicted_idx = torch.max(probabilities, 0)
                predicted_class = CLASS_NAMES[predicted_idx.item()]
                
            # 5. Salva i dati della predizione nel database
            prediction_data = { 
                'class': predicted_class,
                'confidence': f"{predicted_prob.item() * 100:.2f}%",
                'timestamp': datetime.datetime.now(),
                'image_filename': unique_filename,
                'user_id': current_user.id,
                'feedback': None,
                'correct_class_feedback': None
            }
            inserted_id = predictions_collection.insert_one(prediction_data).inserted_id

            # 6. Restituisci la risposta JSON
            return jsonify({
                'class': predicted_class, 
                'confidence': f"{predicted_prob.item() * 100:.2f}%",
                'prediction_id': str(inserted_id)
            })
        except Exception as e:
            # 7. Gestione degli errori, in caso di problemi con il modello o con il salvataggio
            print(f"Si è verificato un errore durante la previsione: {e}")
            return jsonify({'error': f'Errore durante la previsione: {str(e)}'}), 500
        
    return jsonify({'error': 'Errore sconosciuto durante la previsione.'}), 500

@app.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    data = request.get_json()
    prediction_id = data.get('prediction_id')
    feedback_type = data.get('feedback_type')
    correct_class = data.get('correct_class')

    if not prediction_id or not feedback_type:
        return jsonify({'error': 'Dati feedback mancanti.'}), 400

    try:
        update_fields = {
            'feedback': feedback_type,
            'feedback_timestamp': datetime.datetime.now()
        }
        if feedback_type == 'incorrect' and correct_class:
            update_fields['correct_class_feedback'] = correct_class

        result = predictions_collection.update_one(
            {'_id': ObjectId(prediction_id), 'user_id': current_user.id},
            {'$set': update_fields}
        )

        if result.matched_count == 0:
            return jsonify({'error': 'Predizione non trovata o non autorizzata.'}), 404
        
        return jsonify({'message': 'Feedback ricevuto e salvato con successo.'}), 200

    except Exception as e:
        print(f"Errore durante il salvataggio del feedback: {e}")
        return jsonify({'error': f'Errore del server durante il salvataggio del feedback: {str(e)}'}), 500

@app.route('/dashboard')
@login_required
def dashboard():
    class_filter = request.args.get('class')
    feedback_filter = request.args.get('feedback')
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    query = {'user_id': current_user.id}

    if class_filter:
        query['class'] = class_filter
    if feedback_filter:
        if feedback_filter == 'incorrect':
            query['feedback'] = 'incorrect'
            query['correct_class_feedback'] = {'$ne': None} 
        else:
            query['feedback'] = feedback_filter
    if start_date_str:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
        query['timestamp'] = {'$gte': start_date} 
    if end_date_str:
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d') + datetime.timedelta(days=1)
        if 'timestamp' in query:
            query['timestamp']['$lt'] = end_date
        else:
            query['timestamp'] = {'$lt': end_date}
            
    predictions = list(predictions_collection.find(query).sort('timestamp', -1))

    class_counts = {class_name: 0 for class_name in CLASS_NAMES}
    for p in predictions:
        if 'class' in p:
            class_counts[p['class']] += 1

    return render_template('dashboard.html', 
                           predictions=predictions, 
                           class_counts=class_counts,
                           class_names=CLASS_NAMES)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
