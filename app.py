# --- Moduli standard ---
import json
import os
import uuid
import datetime
from datetime import timedelta

# --- Flask e estensioni ---
from flask import (
    Flask, request, jsonify, render_template, redirect, url_for,
    flash, session, send_from_directory, abort
)
from flask_cors import CORS
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, BooleanField
from wtforms.validators import DataRequired

# --- PyTorch ---
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

# --- MongoDB ---
from pymongo import MongoClient
from bson.objectid import ObjectId


# ============== Setup base ==============
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'DEV_KEY_CHANGE_ME')

# Limite dimensione upload (facoltativo): 10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Cookie & session policy
app.config.update(
    REMEMBER_COOKIE_DURATION=timedelta(days=14),      # "Ricordami" 14 giorni
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=45), # Timeout inattività 45 min
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,       # True in produzione (HTTPS)
    REMEMBER_COOKIE_HTTPONLY=True,
    REMEMBER_COOKIE_SECURE=False,      # True in produzione (HTTPS)
)

UPLOAD_FOLDER = 'uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CORS(app)
csrf = CSRFProtect(app)

# --- Flask-Login ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Mongo ---
client = MongoClient('mongodb://localhost:27017/')
db = client.smart_waste
predictions_collection = db.predictions
users_collection = db.users


# ============== Forms ==============
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Registrati')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Ricordami')
    submit = SubmitField('Accedi')

class UploadForm(FlaskForm):
    file = FileField('Seleziona Immagine', validators=[DataRequired()])
    submit = SubmitField('Prevedi')


# ============== User model ==============
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.password_hash = user_data['password']

@login_manager.user_loader
def load_user(user_id):
    doc = users_collection.find_one({"_id": ObjectId(user_id)})
    return User(doc) if doc else None


# ============== Modello ML ==============
CLASS_NAMES = ['plastica', 'carta', 'vetro', 'organico', 'indifferenziato']
NUM_CLASSES = len(CLASS_NAMES)
device = torch.device("cpu")
MODEL_PATH = 'best_model_finetuned_light.pth'

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# torch.load sicuro con fallback
try:
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
except TypeError:
    # Per versioni PyTorch < 2.4
    state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)

# Temperature scaling (se presente)
TEMPERATURE = 1.0
try:
    with open('calibration.json', 'r') as f:
        CALIB = json.load(f)
        TEMPERATURE = float(CALIB.get('temperature', 1.0))
    print(f"Temperature scaling attivo: T={TEMPERATURE}")
except Exception:
    print("Nessuna calibrazione trovata. Uso T=1.0")

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============== Middleware ==============
# Idle timeout
@app.before_request
def idle_logout():
    if not current_user.is_authenticated:
        return
    now_ts = datetime.datetime.utcnow().timestamp()
    last = session.get('last_seen', now_ts)
    if now_ts - last > app.config['PERMANENT_SESSION_LIFETIME'].total_seconds():
        logout_user()
        session.clear()
        flash('Sessione scaduta per inattività. Accedi di nuovo.', 'info')
        return redirect(url_for('login'))
    session['last_seen'] = now_ts

# Blocca la navigazione con predizione pendente
# → CONSENTE solo: app, predict, feedback, upload file, auth, static.
@app.before_request
def force_feedback_before_navigation():
    if not current_user.is_authenticated:
        return
    pending_id = session.get('pending_prediction_id')
    if not pending_id:
        return

    allowed_endpoints = {
        'app', 'predict', 'submit_feedback', 'uploaded_file',
        'login', 'register', 'logout', 'static'
    }
    endpoint = (request.endpoint or '')
    if request.method == 'GET' and endpoint not in allowed_endpoints:
        flash('Conferma prima la predizione in corso per proseguire.', 'error')
        return redirect(url_for('app'))

# Helper per {{ now() }} nei template
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now}


# ============== Routes ==============
@app.route('/')
def landing():
    # Avviso di eventuali vecchie pending se non c'è una pending in sessione
    pending_count = 0
    if current_user.is_authenticated and not session.get('pending_prediction_id'):
        pending_count = predictions_collection.count_documents({
            'user_id': current_user.id,
            'feedback': None
        })
    return render_template('landing.html', pending_count=pending_count)

@app.route('/app', endpoint='app')
@login_required
def classify():
    form = UploadForm()
    preload = None
    pending_id = session.get('pending_prediction_id')
    if pending_id:
        doc = predictions_collection.find_one({
            '_id': ObjectId(pending_id),
            'user_id': current_user.id
        })
        if doc:
            preload = {
                'id': str(doc['_id']),
                'cls': doc.get('class'),
                'conf': doc.get('confidence'),
                'img': doc.get('image_filename')
            }
    return render_template('index.html', form=form,
                           class_names=CLASS_NAMES, preload=preload)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('app'))
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        password = form.password.data
        if users_collection.find_one({'username': username}):
            flash('Username già esistente.', 'error')
            return render_template('register.html', form=form)
        users_collection.insert_one({
            'username': username,
            'password': generate_password_hash(password)
        })
        flash('Registrazione completata! Ora accedi.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('app'))
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        password = form.password.data
        user_data = users_collection.find_one({'username': username})
        if user_data and check_password_hash(user_data['password'], password):
            login_user(User(user_data), remember=form.remember.data)
            session.permanent = True
            session['last_seen'] = datetime.datetime.utcnow().timestamp()
            flash('Login effettuato!', 'success')
            return redirect(url_for('app'))
        flash('Credenziali errate.', 'error')
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    logout_user()
    session.clear()
    flash('Logout effettuato.', 'info')
    return redirect(url_for('landing'))

@app.route('/dashboard')
@login_required
def dashboard():
    query = {'user_id': current_user.id}
    predictions = list(predictions_collection.find(query).sort('timestamp', -1))
    class_counts = {c: 0 for c in CLASS_NAMES}
    pending_exists = any(p.get('feedback') is None for p in predictions)
    for p in predictions:
        if 'class' in p:
            class_counts[p['class']] += 1
    return render_template(
        'dashboard.html',
        predictions=predictions,
        class_names=CLASS_NAMES,
        class_counts=class_counts,
        pending_exists=pending_exists
    )

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'Nessun file inviato.'}), 400

    file = request.files['file']
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)

    try:
        file.save(filepath)
        # Validazione base immagine
        Image.open(filepath).verify()
        img = Image.open(filepath).convert('RGB')
    except (UnidentifiedImageError, OSError):
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': 'File non valido o danneggiato.'}), 400

    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        # ---- Calibrazione confidenza ----
        if TEMPERATURE and TEMPERATURE != 1.0:
            output = output / TEMPERATURE
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_prob, predicted_idx = torch.max(probabilities, 0)
        predicted_class = CLASS_NAMES[predicted_idx.item()]

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
    session['pending_prediction_id'] = str(inserted_id)

    return jsonify({
        'class': predicted_class,
        'confidence': f"{predicted_prob.item() * 100:.2f}%",
        'prediction_id': str(inserted_id)
    })

# ---- FEEDBACK: esente da CSRF ----
@app.route('/feedback', methods=['POST'])
@login_required
@csrf.exempt
def submit_feedback():
    data = request.get_json(silent=True) or {}
    prediction_id = data.get('prediction_id')
    feedback_type = data.get('feedback_type')
    correct_class = data.get('correct_class')

    if not prediction_id or not feedback_type:
        return jsonify({'error': 'Dati mancanti.'}), 400

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
        return jsonify({'error': 'Predizione non trovata.'}), 404

    if session.get('pending_prediction_id') == prediction_id:
        session.pop('pending_prediction_id', None)

    return jsonify({'message': 'Feedback salvato.'}), 200

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    pred = predictions_collection.find_one({'image_filename': filename, 'user_id': current_user.id})
    if not pred:
        abort(403)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ============== Run ==============
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
