# --- Moduli standard ---
import os
import io
import csv
import hmac
import json
import uuid
import shutil
import hashlib
import tempfile
import datetime
import zipfile
from pathlib import Path
from datetime import timedelta
from functools import wraps
from typing import Optional


# --- Flask e estensioni ---
from flask import (
    Flask, request, jsonify, render_template, redirect, url_for,
    flash, session, send_from_directory, abort, send_file
)
from flask_cors import CORS
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
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


# ========================== Setup base ==========================
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'DEV_KEY_CHANGE_ME')

# Limite dimensione upload (10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Consenti CSRF via header per l'AJAX della pagina (fetch con "X-CSRFToken")
app.config['WTF_CSRF_HEADERS'] = ['X-CSRFToken', 'X-CSRF-Token']

# Cookie & session policy
app.config.update(
    REMEMBER_COOKIE_DURATION=timedelta(days=14),       # "Ricordami" 14 giorni
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=45),  # Timeout inattività 45 min
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,                       # True in produzione (HTTPS)
    REMEMBER_COOKIE_HTTPONLY=True,
    REMEMBER_COOKIE_SECURE=False,                      # True in produzione (HTTPS)
)

UPLOAD_FOLDER = 'uploaded_images'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tipi consentiti e mapping estensione
ALLOWED_MIME = {'image/jpeg', 'image/png', 'image/webp'}
MIME_TO_EXT = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/webp': '.webp'
}

CORS(app)
csrf = CSRFProtect(app)

# --- Handler 413: file troppo grande ---
@app.errorhandler(RequestEntityTooLarge)
def handle_413(_e):
    return jsonify({'error': 'File troppo grande (limite massimo superato).'}), 413


# ========================== Flask-Login ==========================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ========================== MongoDB ==========================
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URL)
db = client.smart_waste
predictions_collection = db.predictions
users_collection = db.users


# ========================== Forms ==========================
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


# ========================== User model + loader ==========================
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.password_hash = user_data['password']
        self.role = user_data.get('role', 'user')  # 'user' | 'admin'

@login_manager.user_loader
def load_user(user_id):
    doc = users_collection.find_one({"_id": ObjectId(user_id)})
    return User(doc) if doc else None


# --- Bootstrap admin opzionale (da .env) ---
# ADMIN_USERNAME=tuonome
# ADMIN_PASSWORD=supersegreta (opzionale: se assente e l'utente esiste, lo promuove; se non esiste, crea con password random)
admin_username = os.environ.get('ADMIN_USERNAME')
admin_password = os.environ.get('ADMIN_PASSWORD')  # opzionale
if admin_username:
    existing = users_collection.find_one({'username': admin_username})
    if existing:
        if existing.get('role') != 'admin':
            users_collection.update_one({'_id': existing['_id']}, {'$set': {'role': 'admin'}})
            print(f"[BOOT] Promosso admin: {admin_username}")
    else:
        pwd = admin_password or uuid.uuid4().hex
        users_collection.insert_one({
            'username': admin_username,
            'password': generate_password_hash(pwd),
            'role': 'admin'
        })
        print(f"[BOOT] Creato admin: {admin_username} (password generata se non fornita)")


# ========================== Modello ML ==========================
CLASS_NAMES = ['plastica', 'carta', 'vetro', 'organico', 'indifferenziato']
NUM_CLASSES = len(CLASS_NAMES)
device = torch.device("cpu")

# Percorsi artefatti modello
MODEL_PATH = os.environ.get('MODEL_PATH', 'best_model_finetuned_light.pth')
CALIB_PATH = os.environ.get('CALIB_PATH', 'calibration.json')

# Costruzione modello (ResNet18 con ultimo layer per 5 classi)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Caricamento pesi con fallback (PyTorch >=2.4 ha weights_only)
try:
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
except TypeError:
    state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)

# Temperature scaling (se presente)
TEMPERATURE = 1.0
try:
    with open(CALIB_PATH, 'r', encoding='utf-8') as f:
        CALIB = json.load(f)
        TEMPERATURE = float(CALIB.get('temperature', 1.0))
    print(f"[CALIB] Temperature scaling attivo: T={TEMPERATURE}")
except Exception:
    print("[CALIB] Nessuna calibrazione trovata. Uso T=1.0")

model.to(device)
model.eval()

# Trasformazioni immagine (coerenti con il training)
transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ========================== Decoratori & helpers ==========================
def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or getattr(current_user, 'role', 'user') != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return wrapper

def _hash_user_id(user_id: str) -> str:
    """Hash HMAC-SHA256 dell'ID utente usando un 'pepper' da .env (USER_HASH_PEPPER)."""
    pepper = os.environ.get('USER_HASH_PEPPER', 'pepper-default-change-me')
    return hmac.new(pepper.encode('utf-8'), str(user_id).encode('utf-8'), hashlib.sha256).hexdigest()

def compute_label_and_weight(doc):
    """
    Determina label_final, label_source, weight per l'export.
    Politica semplice:
    - feedback == 'correct'                     -> label_final = predetta,        source='user_verified',  weight=1.0
    - feedback == 'incorrect' + correzione     -> label_final = correzione,      source='user_corrected', weight=0.9
    - altrimenti (non verificata)              -> label_final = predetta,        source='unverified',     weight=0.3
    """
    predicted = doc.get('class', '')
    feedback  = (doc.get('feedback') or '').strip().lower()
    corrected = (doc.get('correct_class_feedback') or '').strip().lower()

    if feedback == 'correct' and predicted in CLASS_NAMES:
        return predicted, 'user_verified', 1.0
    if feedback == 'incorrect' and corrected in CLASS_NAMES:
        return corrected, 'user_corrected', 0.9
    if predicted in CLASS_NAMES:
        return predicted, 'unverified', 0.3
    return '', 'unverified', 0.3

def materialize_image_local(doc, tmp_images_root: Path) -> Optional[Path]:
    """
    Copia l'immagine originale (salvata in UPLOAD_FOLDER) in una cartella temporanea e
    ritorna il Path alla copia. Se manca, ritorna None.
    """
    fname = doc.get('image_filename')
    if not fname:
        return None
    src = Path(app.config['UPLOAD_FOLDER']) / fname
    if not src.exists():
        return None
    dst = tmp_images_root / fname
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


# ========================== Middleware ==========================
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

# Blocca navigazione con predizione pendente (consente: app, predict, feedback, upload, auth, static, dashboard)
@app.before_request
def force_feedback_before_navigation():
    if not current_user.is_authenticated:
        return
    pending_id = session.get('pending_prediction_id')
    if not pending_id:
        return
    allowed_endpoints = {'app', 'predict', 'submit_feedback', 'uploaded_file',
                         'login', 'register', 'logout', 'static', 'dashboard'}
    endpoint = (request.endpoint or '')
    if request.method == 'GET' and endpoint not in allowed_endpoints:
        flash('Conferma prima la predizione in corso per proseguire.', 'error')
        return redirect(url_for('app'))

# Helper per template
@app.context_processor
def inject_now():
    return {
        'now': datetime.datetime.now,
        'is_admin': (getattr(current_user, 'role', 'user') == 'admin') if current_user.is_authenticated else False
    }


# ========================== Routes ==========================
@app.route('/')
def landing():
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
    return render_template('index.html', form=form, class_names=CLASS_NAMES, preload=preload)


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
            'password': generate_password_hash(password),
            'role': 'user'
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
    """
    Riceve un file immagine via form-data -> salva -> valida -> inferenza -> salva esito in Mongo -> ritorna JSON.
    """
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'Nessun file inviato.'}), 400

    file = request.files['file']

    # Validazione tipo MIME
    mime = file.mimetype or ''
    if mime not in ALLOWED_MIME:
        return jsonify({'error': 'Formato non supportato. Usa JPG/PNG/WebP.'}), 400

    # Costruzione nome sicuro e path destinazione
    # Se l'utente carica "senza estensione", usiamo quella dedotta dal MIME
    base_name = secure_filename(file.filename) or f"upload_{uuid.uuid4().hex}"
    root, ext = os.path.splitext(base_name)
    if ext.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
        ext = MIME_TO_EXT.get(mime, '.jpg')
    unique_filename = f"{uuid.uuid4().hex}_{root}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    # Salvataggio + validazione immagine
    try:
        file.save(filepath)
        Image.open(filepath).verify()  # quick sanity check
        img = Image.open(filepath).convert('RGB')
    except (UnidentifiedImageError, OSError):
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': 'File non valido o danneggiato.'}), 400

    # Inferenza
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        if TEMPERATURE and TEMPERATURE != 1.0:
            logits = logits / TEMPERATURE
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)
        prob_val, pred_idx = torch.max(probabilities, 0)
        predicted_class = CLASS_NAMES[pred_idx.item()]

    prob_float = float(prob_val.item())

    # Persistenza
    prediction_data = {
        'class': predicted_class,
        'confidence': f"{prob_float * 100:.2f}%",
        'prob': prob_float,  # utile per filtri export
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
        'confidence': f"{prob_float * 100:.2f}%",
        'prediction_id': str(inserted_id)
    })


# ---- FEEDBACK (CSRF esente) ----
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


# ========================== EXPORT (SOLO ADMIN) ==========================
@app.route('/export', methods=['GET'])
@login_required
@admin_required
def export_page():
    return render_template('export.html', class_names=CLASS_NAMES)

@app.route('/export', methods=['POST'])
@login_required
@admin_required
def do_export():
    """
    Esporta ZIP con:
      /dataset.csv
      /images/<classe>/<filename>
    con pesi/etichette calcolati da compute_label_and_weight().
    """
    include_unverified = request.form.get('include_unverified') == 'on'
    include_user_hash  = request.form.get('include_user_hash') == 'on'
    scope_all          = request.form.get('scope') == 'all'  # 'all' o 'mine'

    try:
        min_conf_filter = float(request.form.get('min_conf', '0') or 0)
    except Exception:
        min_conf_filter = 0.0

    date_from_str = (request.form.get('date_from') or '').strip()
    date_to_str   = (request.form.get('date_to') or '').strip()

    q: dict = {}
    if not scope_all:
        q['user_id'] = current_user.id

    if date_from_str:
        try:
            dt = datetime.datetime.fromisoformat(date_from_str)
            q.setdefault('timestamp', {})['$gte'] = dt
        except Exception:
            pass
    if date_to_str:
        try:
            dt = datetime.datetime.fromisoformat(date_to_str)
            q.setdefault('timestamp', {})['$lte'] = dt
        except Exception:
            pass

    preds = list(predictions_collection.find(q).sort('timestamp', -1))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        images_root = tmp / "images"
        images_root.mkdir(parents=True, exist_ok=True)

        rows = []
        copied = 0
        skipped_missing = 0

        for d in preds:
            # filtra per confidenza minima (se presente)
            if d.get('prob') is not None and float(d.get('prob')) < min_conf_filter:
                continue

            label_final, label_source, weight = compute_label_and_weight(d)

            # se non verificata e non richiesta -> skip
            if (label_source == 'unverified') and not include_unverified:
                continue

            # copia immagine
            local_file = materialize_image_local(d, images_root)
            if not local_file or not local_file.exists():
                skipped_missing += 1
                continue

            # sotto-cartella = classe finale (o 'unverified')
            sub = label_final if label_final in CLASS_NAMES else 'unverified'
            target_dir = images_root / sub
            target_dir.mkdir(exist_ok=True, parents=True)
            final_path = target_dir / local_file.name
            shutil.move(str(local_file), str(final_path))
            copied += 1

            # riga CSV
            row = {
                'id': str(d.get('_id')),
                'filename': final_path.name,
                'label_final': label_final,
                'label_source': label_source,
                'weight': f"{weight:.4f}",
                'model_confidence': f"{float(d.get('prob') or 0.0):.6f}",
                'feedback': d.get('feedback') or '',
                'correct_class_feedback': d.get('correct_class_feedback') or '',
                'timestamp': (d.get('timestamp') or datetime.datetime.utcnow()).isoformat()
            }
            if include_user_hash:
                row['user_hash'] = _hash_user_id(d.get('user_id'))

            rows.append(row)

        # CSV
        csv_path = tmp / "dataset.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'id','filename','label_final','label_source','weight',
                'model_confidence','feedback','correct_class_feedback','timestamp'
            ]
            if include_user_hash:
                fieldnames.append('user_hash')
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # ZIP in memoria
        zip_name = f"smart_waste_export_{datetime.datetime.utcnow():%Y%m%d_%H%M%S}.zip"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(csv_path, arcname="dataset.csv")
            for p in images_root.rglob('*'):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(tmp)))
        buffer.seek(0)

        print(f"[EXPORT] copiati: {copied}, mancanti: {skipped_missing}, righe CSV: {len(rows)}")
        return send_file(buffer, as_attachment=True, download_name=zip_name, mimetype='application/zip')


# ========================== Run ==========================
if __name__ == '__main__':
    # In produzione: host='0.0.0.0', debug=False
    app.run(debug=True, host='127.0.0.1', port=5000)
