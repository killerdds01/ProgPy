# === Smart Waste · Flask App (FINAL) ==========================================
# Requisiti: Flask, Flask-Login, Flask-WTF, flask-cors, python-dotenv,
#            pymongo, pillow, torch, torchvision, email-validator
# Opzionale: Flask-Limiter (se assente, i limit diventano no-op)
# ==============================================================================
#
# COME SI USA (sviluppo):
#   1) Crea e attiva un virtualenv e installa i requirement (requirements.txt).
#   2) Assicurati che MongoDB sia raggiungibile. In locale puoi:
#        - farlo partire tu (mongod), oppure
#        - impostare MONGO_AUTOSTART=true e (opzionale) MONGO_BIN per l'auto-avvio.
#   3) Avvia:     python app.py
#   4) Il terminale stamperà gli URL per collegarti da:
#        - PC:  http://127.0.0.1:5000/app
#        - LAN: http://<IP_DEL_PC>:5000/app  (telefono/altro PC sulla stessa Wi-Fi)
#
# VARIABILI UTILI (.env facoltativo):
#   SECRET_KEY=qualcosa-di-segreto
#   FLASK_RUN_HOST=0.0.0.0         # ascolta su tutta la LAN (default di questo file)
#   FLASK_RUN_PORT=5000
#   MONGO_URL=mongodb://127.0.0.1:27017/
#   MONGO_AUTOSTART=false|true     # true = prova ad avviare mongod in dev
#   MONGO_BIN=percorso\mongod.exe  # se vuoi forzare il binario
#   MODEL_PATH=best_model_finetuned_light.pth
#   PASSWORD_PEPPER=pepper-per-hash-password
#   USER_HASH_PEPPER=pepper-per-hash-user (export admin)
#
# SICUREZZA (sviluppo vs produzione):
#   - Questo è il server di sviluppo di Flask (comodo per test, NON usare in produzione).
#   - In produzione usa un WSGI (gunicorn/uwsgi) dietro un reverse proxy HTTPS,
#     e imposta i cookie 'SECURE' a True.
# ==============================================================================

# --- Stdlib -------------------------------------------------------------------
import os, io, csv, hmac, json, uuid, time, shutil, zipfile, datetime, tempfile, subprocess, hashlib, math, socket
from pathlib import Path
from typing import Optional
from datetime import timedelta

# --- Flask & estensioni -------------------------------------------------------
from flask import (
    Flask, request, jsonify, render_template, redirect, url_for,
    flash, session, send_from_directory, abort, send_file
)
from flask_cors import CORS
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect, CSRFError, generate_csrf
from wtforms import StringField, PasswordField, SubmitField, FileField, BooleanField
from wtforms.validators import DataRequired, Email, Length, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ---- Rate-limit opzionale ----------------------------------------------------
# Se Flask-Limiter non è installato, creiamo un "finto" decoratore .limit()
# in modo che il codice giri ugualmente (no-op). Questo evita if/else dappertutto.
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    _HAS_LIMITER = True
except Exception:
    _HAS_LIMITER = False
    class _DummyLimiter:
        def __init__(self, *a, **k): pass
        def limit(self, *a, **k):
            def deco(f): return f
            return deco
    Limiter = _DummyLimiter        # type: ignore
    def get_remote_address(): return None  # type: ignore

# --- PyTorch ------------------------------------------------------------------
# Carichiamo un modello ResNet18 già "finetunato" per 5 classi di rifiuti.
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError, ImageOps  # ImageOps per exif_transpose

# (Opzionale) Supporto HEIC/HEIF se pillow-heif è installato (foto iPhone, ecc.)
# Se manca, non blocchiamo l'app: semplicemente rifiutiamo HEIC/HEIF con 415.
try:
    import pillow_heif  # pip install pillow-heif
    pillow_heif.register_heif_opener()
    _HAS_HEIF = True
except Exception:
    _HAS_HEIF = False

# --- MongoDB ------------------------------------------------------------------
from pymongo import MongoClient
from bson.objectid import ObjectId

# --- Env ----------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# === Flask base ===============================================================
# static_folder: puntiamo a ./static/static così i template possono usare /static/...
# NB: Tutto in static/ è pubblico e servito così com'è (soggetto a CSP).
app = Flask(
    __name__,
    static_folder=os.path.join('static', 'static'),
    static_url_path='/static',
    template_folder='templates'
)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'DEV_KEY_CHANGE_ME')

# Limite upload 10MB (Flask risponde 413 automaticamente se superi)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Cookie & sessione (in produzione imposta i flag SECURE=True dietro HTTPS!)
# PERMANENT_SESSION_LIFETIME: usato anche dall'idle_logout per scadenza inattività.
app.config.update(
    REMEMBER_COOKIE_DURATION=timedelta(days=14),
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=45),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,   # True in produzione (HTTPS)
    REMEMBER_COOKIE_HTTPONLY=True,
    REMEMBER_COOKIE_SECURE=False,  # True in produzione (HTTPS)
)

# Limiter (se il pacchetto non c'è, è un no-op)
# default_limits applicato se il pacchetto è presente.
limiter = Limiter(key_func=get_remote_address, app=app, default_limits=["200/hour"])

# Cartelle di storage (create se non esistono)
UPLOAD_FOLDER  = 'uploaded_images'
PROFILE_FOLDER = 'profile_photos'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(PROFILE_FOLDER).mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER']  = UPLOAD_FOLDER
app.config['PROFILE_FOLDER'] = PROFILE_FOLDER

# Tipi MIME supportati per le immagini e mappatura estensione file.
# Usiamo le estensioni derivate dal MIME per evitare estensioni "fake".
ALLOWED_MIME = {'image/jpeg', 'image/png', 'image/webp'}
MIME_TO_EXT  = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp'}

# Se supporto HEIC attivo, aggiungiamo i MIME e salviamo come .jpg per coerenza col modello.
if _HAS_HEIF:
    ALLOWED_MIME = set(ALLOWED_MIME) | {'image/heic', 'image/heif'}
    MIME_TO_EXT.update({'image/heic': '.jpg', 'image/heif': '.jpg'})

# Scartiamo immagini troppo piccole (anti-micro-immagini)
MIN_IMG_W = 96
MIN_IMG_H = 96

# CORS & CSRF (classiche protezioni per form e fetch)
CORS(app)
csrf = CSRFProtect(app)

# === Login manager ============================================================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # se non autenticato, redirect qui

# === MongoDB (autostart dev) ==================================================
# Se MONGO_AUTOSTART=true prova ad avviare mongod localmente in dev.
MONGO_URL    = os.environ.get('MONGO_URL', 'mongodb://127.0.0.1:27017/')
MONGO_PORT   = int(os.environ.get('MONGO_PORT', '27017') or 27017)
MONGO_AUTO   = (os.environ.get('MONGO_AUTOSTART', 'false').lower() == 'true')
MONGO_BIN    = os.environ.get('MONGO_BIN', '').strip().strip('"').strip("'")
MONGO_DBPATH = os.environ.get('MONGO_DBPATH', './mongo_data')

client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=2000, connectTimeoutMS=2000)

def _mongo_up() -> bool:
    """True se MongoDB risponde al ping, altrimenti False (usato a bootstrap e runtime)."""
    try:
        client.admin.command('ping')
        return True
    except Exception as e:
        print(f"[WARN] MongoDB non raggiungibile: {e}")
        return False

def _try_start_mongo() -> bool:
    """
    Avvia mongod localmente (solo dev) se abilitato via env e non è up.
    Per Windows passa flag per staccare il processo dalla console corrente.
    """
    if not MONGO_AUTO or _mongo_up():
        return _mongo_up()
    mongod = MONGO_BIN or shutil.which("mongod")
    if not mongod or not Path(mongod).exists():
        print(f"[MONGO] mongod non trovato. MONGO_BIN='{os.environ.get('MONGO_BIN','')}'.")
        return False
    Path(MONGO_DBPATH).mkdir(parents=True, exist_ok=True)
    args = [mongod, "--dbpath", str(MONGO_DBPATH), "--port", str(MONGO_PORT), "--bind_ip", "127.0.0.1", "--quiet"]
    flags = 0
    if os.name == "nt":  # DETACHED + NEW_PROCESS_GROUP su Windows
        flags = 0x00000200 | 0x00000008
    try:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        time.sleep(2.0)
        if _mongo_up():
            print(f"[MONGO] mongod avviato su 127.0.0.1:{MONGO_PORT} (pid={proc.pid})")
            return True
    except Exception as e:
        print(f"[MONGO] Avvio mongod fallito: {e}")
    return False

# Flag + handle collezioni solo se DB è pronto (protezione dai None nel codice sotto)
DB_READY = _mongo_up() or (_try_start_mongo() and _mongo_up())
db  = client.smart_waste if DB_READY else None
predictions_collection = db.predictions if DB_READY else None
users_collection       = db.users if DB_READY else None

# === Modello ==================================================================
# Classi previste dal modello (in ordine stabile)
CLASS_NAMES = ['plastica', 'carta', 'vetro', 'organico', 'indifferenziato']

# Usiamo CPU per semplicità/portabilità (niente dipendenze CUDA nel deploy base)
device = torch.device("cpu")
MODEL_PATH = os.environ.get('MODEL_PATH', 'best_model_finetuned_light.pth')

# ResNet18 con testa a N classi (carichiamo solo i pesi, non le weight predefinite)
# Perché weights=None? Perché carichiamo DI SEGUITO i nostri pesi finetunati.
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
try:
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)  # PyTorch >= 2.0
except TypeError:
    state = torch.load(MODEL_PATH, map_location=device)                     # fallback PyTorch < 2.0
model.load_state_dict(state)
model.to(device).eval()  # eval(): no dropout, BN in inference

# Temperature scaling (affina le probabilità senza ri-addestrare)
TEMPERATURE = 1.0
try:
    with open('calibration.json','r',encoding='utf-8') as f:
        TEMPERATURE = float(json.load(f).get('temperature',1.0))
    print(f"[CALIB] T={TEMPERATURE}")
except Exception:
    print("[CALIB] T=1.0")

# Trasformazioni immagine (devono essere le stesse usate in training/validazione)
transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# === Forms ====================================================================
# Nota: Flask-WTF gestisce CSRF e validazioni per i form HTML.
class RegistrationForm(FlaskForm):
    username   = StringField('Username (opzionale)')
    first_name = StringField('Nome',     validators=[DataRequired(), Length(min=1, max=60)])
    last_name  = StringField('Cognome',  validators=[DataRequired(), Length(min=1, max=60)])
    email      = StringField('Email',    validators=[DataRequired(), Email()])
    password   = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm    = PasswordField('Conferma password', validators=[DataRequired(), EqualTo('password')])
    consent_training_default = BooleanField('Acconsento all’uso delle immagini per addestrare il modello.')
    accept_tos = BooleanField('Accetto la privacy policy', validators=[DataRequired()])
    submit     = SubmitField('Registrati')

class LoginForm(FlaskForm):
    username = StringField('Email o Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Ricordami')
    submit   = SubmitField('Accedi')

class UploadForm(FlaskForm):
    file   = FileField('Seleziona Immagine', validators=[DataRequired()])
    submit = SubmitField('Prevedi')

# === Helpers ==================================================================
def _peppered(pw: str) -> str:
    """Aggiunge un 'pepper' (dal .env) alla password prima dell'hash per più sicurezza."""
    return f"{os.environ.get('PASSWORD_PEPPER','')}{pw}"

class User(UserMixin):
    """Wrapper per Flask-Login: trasforma il documento Mongo in un oggetto 'User'."""
    def __init__(self, d):
        self.id = str(d['_id'])
        self.username = d.get('username') or d.get('email')
        self.email = d.get('email')
        self.first_name = d.get('first_name')
        self.last_name  = d.get('last_name')
        self.role = d.get('role','user')
        self.consent_training_default = bool(d.get('consent_training_default', False))
        self.profile_image = d.get('profile_image')

@login_manager.user_loader
def load_user(user_id):
    """Carica l'utente per la sessione (dato l'id in sessione)."""
    if not DB_READY: return None
    doc = users_collection.find_one({"_id": ObjectId(user_id)})
    return User(doc) if doc else None

# Bootstrap admin opzionale all'avvio (solo se ADMIN_USERNAME è presente)
# Perché: per avere almeno un admin al primo run senza toccare il DB a mano.
admin_username = os.environ.get('ADMIN_USERNAME')
admin_password = os.environ.get('ADMIN_PASSWORD')
if admin_username and DB_READY:
    try:
        existing = users_collection.find_one({'username': admin_username})
        if existing:
            if existing.get('role') != 'admin':
                users_collection.update_one({'_id': existing['_id']},{'$set':{'role':'admin'}})
                print(f"[BOOT] Promosso admin: {admin_username}")
        else:
            pwd = admin_password or uuid.uuid4().hex
            users_collection.insert_one({
                'username': admin_username, 'email': None,
                'first_name':'Admin','last_name':'',
                'password': generate_password_hash(_peppered(pwd)),
                'role':'admin','consent_training_default':False,'profile_image':None
            })
            print(f"[BOOT] Creato admin: {admin_username}")
    except Exception as e:
        print(f"[WARN] Bootstrap admin saltato: {e}")

def compute_label_and_weight(doc):
    """
    Stabilisce l'etichetta finale e un peso per il training export:
      - feedback == 'correct'                -> (predetta, 'user_verified',   1.0)
      - feedback == 'incorrect' + correzione -> (correzione, 'user_corrected',0.9)
      - altrimenti (non verificata)          -> (predetta, 'unverified',      0.3)
    Razionale: punire un po' (0.9) i casi corretti dall'utente e meno (0.3) i non verificati.
    """
    pred = doc.get('class','')
    fb   = (doc.get('feedback') or '').strip().lower()
    corr = (doc.get('correct_class_feedback') or '').strip().lower()
    if fb == 'correct' and pred in CLASS_NAMES:     return pred,'user_verified',1.0
    if fb == 'incorrect' and corr in CLASS_NAMES:   return corr,'user_corrected',0.9
    if pred in CLASS_NAMES:                         return pred,'unverified',0.3
    return '','unverified',0.3

def materialize_image_local(doc, root: Path) -> Optional[Path]:
    """
    Copia il file immagine di una predizione in una cartella temporanea (per export).
    Ritorna il percorso di destinazione se esiste, altrimenti None.
    """
    fname = doc.get('image_filename')
    if not fname: return None
    src = Path(app.config['UPLOAD_FOLDER'])/fname
    if not src.exists(): return None
    dst = root/fname
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src,dst)
    return dst

# === Fiducia utente (Admin) ===================================================
def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Lower bound dell'intervallo di Wilson per proporzioni (stima prudente)."""
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1 + z*z/total
    centre = phat + (z*z)/(2*total)
    margin = z * math.sqrt((phat*(1-phat) + (z*z)/(4*total)) / total)
    return max(0.0, (centre - margin) / denom)

def compute_user_trust(user_id: str) -> int:
    """
    Calcola un punteggio di fiducia 0–100:
      - accuratezza verificata (Wilson LB)
      - quanto spesso verifica (verify_rate)
      - quanto spesso consente l'uso per training (consent_rate)
    Pesi: 0.65, 0.20, 0.15 — conservativi per privilegiare la qualità dei feedback.
    """
    total = predictions_collection.count_documents({'user_id': user_id})
    if total == 0:
        return 0
    n_correct   = predictions_collection.count_documents({'user_id': user_id, 'feedback':'correct'})
    n_incorrect = predictions_collection.count_documents({'user_id': user_id, 'feedback':'incorrect'})
    verified = n_correct + n_incorrect
    correct_lb = wilson_lower_bound(n_correct, verified) if verified else 0.0
    verify_rate = verified / total
    consent_rate = predictions_collection.count_documents({'user_id': user_id, 'consent_training': True}) / total
    trust = (0.65 * correct_lb + 0.20 * verify_rate + 0.15 * consent_rate) * 100.0
    return int(round(max(0.0, min(100.0, trust))))

# === Middleware & headers =====================================================
@app.before_request
def idle_logout():
    """
    Se l'utente è inattivo oltre PERMANENT_SESSION_LIFETIME, si forza il logout.
    Evita che sessioni lasciate aperte restino attive per sempre.
    """
    if not current_user.is_authenticated: return
    now = datetime.datetime.utcnow().timestamp()
    last = session.get('last_seen', now)
    if now - last > app.config['PERMANENT_SESSION_LIFETIME'].total_seconds():
        logout_user(); session.clear()
        flash('Sessione scaduta per inattività. Accedi di nuovo.', 'info')
        return redirect(url_for('login'))
    session['last_seen'] = now

@app.before_request
def require_db_if_needed():
    """
    Blocca l'accesso alle pagine che richiedono DB quando il DB non è pronto.
    Le uniche pagine accessibili in tal caso sono landing/login/register/static.
    Razionale: UX chiara quando il DB è giù (dev o failure).
    """
    if DB_READY: return
    if (request.endpoint or '') not in {'landing','login','register','static'}:
        return render_template("error.html", code=503, message="Database offline. Avvia MongoDB e ricarica."), 503

@app.before_request
def force_feedback_before_navigation():
    """
    Se esiste una predizione 'pending' in sessione, costringe a dare feedback
    prima di girare il resto dell'app. Evita code di predizioni mai confermate.
    """
    if not current_user.is_authenticated: return
    if not session.get('pending_prediction_id'): return
    allowed={'app','predict','submit_feedback','uploaded_file','login','register','logout','static',
             'dashboard','profile','profile_photo_upload','profile_photo_get','delete_prediction',
             'account_delete','profile_password','account_export','export_page','do_export',
             'admin_trust','admin_trust_json'}
    if request.method=='GET' and (request.endpoint or '') not in allowed:
        flash('Conferma prima la predizione in corso per proseguire.', 'error')
        return redirect(url_for('app'))

@app.after_request
def set_security_headers(resp):
    """
    Applica una Content-Security-Policy di base e altri header di sicurezza.
    In produzione puoi irrigidire ulteriormente la CSP.
    """
    csp=("default-src 'self'; img-src 'self' data: blob:; "
         "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
         "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
         "connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'self'")
    resp.headers['Content-Security-Policy']=csp
    resp.headers['X-Content-Type-Options']='nosniff'
    resp.headers['Referrer-Policy']='strict-origin-when-cross-origin'
    resp.headers['Permissions-Policy']='camera=(), microphone=(), geolocation=()'
    return resp

@app.context_processor
def inject_now():
    """Rende disponibili nei template: ora corrente, flag admin, helper CSRF."""
    return {'now': datetime.datetime.now,
            'is_admin': (getattr(current_user,'role','user')=='admin') if current_user.is_authenticated else False,
            'csrf_token': generate_csrf}

# === Error pages (usa templates/error.html) ===================================
@app.errorhandler(CSRFError)
def err_csrf(e): return render_template("error.html", code=400, message="Token CSRF non valido"), 400
@app.errorhandler(400)
def err_400(e): return render_template("error.html", code=400, message="Richiesta non valida"), 400
@app.errorhandler(401)
def err_401(e): return render_template("error.html", code=401, message="Non autenticato"), 401
@app.errorhandler(403)
def err_403(e): return render_template("error.html", code=403, message="Accesso negato"), 403
@app.errorhandler(404)
def err_404(e): return render_template("error.html", code=404, message="Risorsa non trovata"), 404
@app.errorhandler(413)
def err_413(e): return render_template("error.html", code=413, message="File troppo grande"), 413
@app.errorhandler(429)
def err_429(e): return render_template("error.html", code=429, message="Troppe richieste"), 429
@app.errorhandler(500)
def err_500(e): return render_template("error.html", code=500, message="Errore interno"), 500
@app.errorhandler(503)
def err_503(e): return render_template("error.html", code=503, message="Servizio non disponibile"), 503

# === Routes: Landing & App ====================================================
@app.route('/')
def landing():
    """
    Pagina di benvenuto. Se l'utente è loggato, mostra quante predizioni pendenti
    (senza feedback) ha nel badge. Questo aiuta a “chiudere il loop” del feedback.
    """
    pending = 0
    if current_user.is_authenticated and not session.get('pending_prediction_id') and DB_READY:
        pending = predictions_collection.count_documents({'user_id': current_user.id, 'feedback': None})
    return render_template('landing.html', pending_count=pending)

@app.route('/app', endpoint='app')
@login_required
def classify():
    """
    Pagina principale di classificazione:
      - upload di un'immagine
      - mostra ultima predizione 'pending' (se esiste) da confermare/correggere
    """
    form = UploadForm()
    preload=None
    if DB_READY:
        pid = session.get('pending_prediction_id')
        if pid:
            d = predictions_collection.find_one({'_id': ObjectId(pid), 'user_id': current_user.id})
            if d:
                preload={'id':str(d['_id']),'cls':d.get('class'),'conf':d.get('confidence'),'img':d.get('image_filename')}
    return render_template('index.html', form=form, class_names=CLASS_NAMES, preload=preload,
                           user_default_consent=bool(getattr(current_user,'consent_training_default',False)))

# === Auth =====================================================================
@app.route('/register', methods=['GET','POST'])
@limiter.limit("10/minute")
def register():
    """Registrazione utente con validazioni basilari e hash + pepper sulla password."""
    form = RegistrationForm()
    if current_user.is_authenticated: return redirect(url_for('app'))
    if request.method=='POST' and form.validate_on_submit():
        if not DB_READY:
            flash('Database offline: impossibile registrare ora.', 'error')
            return render_template('register.html', form=form)
        email=form.email.data.strip().lower(); username=(form.username.data or '').strip()
        if users_collection.find_one({'email': email}):
            flash('Email già registrata.', 'error')
            return render_template('register.html', form=form)
        users_collection.insert_one({
            'username': username or email.split('@')[0],
            'first_name': form.first_name.data.strip(),
            'last_name' : form.last_name.data.strip(),
            'email': email,
            'password': generate_password_hash(_peppered(form.password.data)),
            'role':'user','created_at': datetime.datetime.utcnow(),
            'consent_training_default': bool(form.consent_training_default.data),
            'profile_image': None
        })
        flash('Registrazione completata! Ora accedi.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET','POST'])
@limiter.limit("10/minute")
def login():
    """Login con email/username + password. Imposta sessione 'permanent'."""
    form = LoginForm()
    if current_user.is_authenticated: return redirect(url_for('app'))
    if request.method=='POST' and form.validate_on_submit():
        if not DB_READY:
            flash('Database offline: impossibile accedere ora.', 'error')
            return render_template('login.html', form=form)
        key=form.username.data.strip()
        q={'$or':[{'email':key.lower()},{'username':key}]}
        u=users_collection.find_one(q)
        if u and check_password_hash(u['password'], _peppered(form.password.data)):
            login_user(User(u), remember=form.remember.data); session.permanent=True
            session['last_seen']=datetime.datetime.utcnow().timestamp()
            flash('Login effettuato!', 'success')
            return redirect(url_for('app'))
        flash('Credenziali errate.', 'error')
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    """Logout, pulizia sessione e redirect alla landing."""
    logout_user(); session.clear(); flash('Logout effettuato.', 'info')
    return redirect(url_for('landing'))

# === Dashboard + filtri =======================================================
@app.route('/dashboard')
@login_required
def dashboard():
    """
    Storico predizioni dell'utente con filtri per classe/stato/intervalli data.
    Nota: i filtri sono in querystring; si può estendere con ulteriori criteri.
    """
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503

    f_class=(request.args.get('class') or '').strip()
    f_state=(request.args.get('state') or 'all').strip()
    f_from =(request.args.get('from') or '').strip()
    f_to   =(request.args.get('to') or '').strip()
    no_images = (request.args.get('no_images') == '1')

    q={'user_id': current_user.id}
    if f_class in CLASS_NAMES: q['class']=f_class
    if f_state=='pending': q['feedback']=None
    elif f_state=='correct': q['feedback']='correct'
    elif f_state=='incorrect': q['feedback']='incorrect'
    if f_from:
        try: q.setdefault('timestamp', {})['$gte']=datetime.datetime.fromisoformat(f_from)
        except Exception: pass
    if f_to:
        try: q.setdefault('timestamp', {})['$lte']=datetime.datetime.fromisoformat(f_to)
        except Exception: pass

    preds=list(predictions_collection.find(q).sort('timestamp',-1))
    class_counts={c:0 for c in CLASS_NAMES}
    pending_exists=any(p.get('feedback') is None for p in preds)
    for p in preds:
        if 'class'in p: class_counts[p['class']]+=1

    return render_template('dashboard.html', predictions=preds, class_names=CLASS_NAMES,
                           class_counts=class_counts, pending_exists=pending_exists,
                           f_class=f_class, f_state=f_state, f_from=f_from, f_to=f_to,
                           no_images=no_images)

# === Profilo ==================================================================
@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    """Pagina profilo: nome/cognome, consenso di default, ecc."""
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    if request.method=='POST':
        first=(request.form.get('first_name') or '').strip()
        last =(request.form.get('last_name') or '').strip()
        cons = True if request.form.get('consent_training_default') in ('on','true','1') else False
        users_collection.update_one({'_id': ObjectId(current_user.id)},
                                    {'$set': {'first_name':first,'last_name':last,'consent_training_default':cons}})
        flash('Profilo aggiornato.', 'success')
        return redirect(url_for('profile'))
    usr=users_collection.find_one({'_id': ObjectId(current_user.id)}) if DB_READY else {}
    return render_template('profile.html', user=usr)

@app.route('/profile/photo', methods=['POST'])
@login_required
@limiter.limit("20/minute")
def profile_photo_upload():
    """Upload foto profilo: ridimensiona e corregge l'orientamento da EXIF."""
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    f = request.files.get('photo')
    if not f or f.filename=='':
        flash('Nessun file selezionato.', 'error')
        return redirect(url_for('profile'))
    if f.mimetype not in ALLOWED_MIME:
        flash('Formato non supportato (JPG/PNG/WebP).', 'error')
        return redirect(url_for('profile'))
    ext=MIME_TO_EXT.get(f.mimetype,'.jpg'); filename=f"profile_{current_user.id}{ext}"
    path=Path(app.config['PROFILE_FOLDER'])/filename
    try:
        tmp=Path(tempfile.mkdtemp())/f"tmp{ext}"; f.save(tmp)
        img=Image.open(tmp)
        try: img = ImageOps.exif_transpose(img)  # auto-rotate EXIF
        except Exception: pass
        img = img.convert('RGB')
        img.thumbnail((512,512))
        img.save(path,quality=88)
        tmp.unlink(missing_ok=True)
        users_collection.update_one({'_id': ObjectId(current_user.id)}, {'$set': {'profile_image': filename}})
        flash('Foto profilo aggiornata.', 'success')
    except Exception:
        flash('Errore durante il salvataggio della foto.', 'error')
    return redirect(url_for('profile'))

@app.route('/profile/photo/<filename>')
@login_required
def profile_photo_get(filename):
    """Ritorna la foto profilo (solo proprietario o admin)."""
    is_owner = filename.startswith(f"profile_{current_user.id}")
    if not is_owner and getattr(current_user,'role','user')!='admin': abort(403)
    return send_from_directory(app.config['PROFILE_FOLDER'], filename)

@app.route('/profile/password', methods=['POST'])
@login_required
@limiter.limit("10/hour")
def profile_password():
    """Cambio password con verifica della vecchia e della conferma."""
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    old=request.form.get('old_password') or ''
    new=request.form.get('new_password') or ''
    conf=request.form.get('confirm_password') or ''
    if len(new)<6 or new!=conf:
        flash('La nuova password deve avere almeno 6 caratteri e coincidere con la conferma.', 'error')
        return redirect(url_for('profile'))
    u=users_collection.find_one({'_id': ObjectId(current_user.id)})
    if not u or not check_password_hash(u['password'], _peppered(old)):
        flash('Password attuale errata.', 'error')
        return redirect(url_for('profile'))
    users_collection.update_one({'_id': ObjectId(current_user.id)},
                                {'$set':{'password': generate_password_hash(_peppered(new))}})
    flash('Password aggiornata correttamente.', 'success')
    return redirect(url_for('profile'))

# === Cancellazione account ====================================================
@app.route('/account/delete', methods=['POST'])
@login_required
@limiter.limit("5/hour")
def account_delete():
    """Elimina account e tutti i dati/immagini associati all'utente."""
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    pwd=request.form.get('password') or ''
    u=users_collection.find_one({'_id': ObjectId(current_user.id)})
    if not u or not check_password_hash(u['password'], _peppered(pwd)):
        flash('Password errata.', 'error')
        return redirect(url_for('profile'))
    # elimina immagini/predizioni
    for p in predictions_collection.find({'user_id': current_user.id}):
        fn=p.get('image_filename')
        if fn:
            fp=Path(app.config['UPLOAD_FOLDER'])/fn
            if fp.exists():
                try: fp.unlink()
                except Exception: pass
    predictions_collection.delete_many({'user_id': current_user.id})
    # elimina foto profilo
    if u.get('profile_image'):
        pp=Path(app.config['PROFILE_FOLDER'])/u['profile_image']
        if pp.exists():
            try: pp.unlink()
            except Exception: pass
    users_collection.delete_one({'_id': ObjectId(current_user.id)})
    logout_user(); session.clear()
    flash('Account eliminato definitivamente.', 'success')
    return redirect(url_for('landing'))

# === Esportazione dati utente (JSON) =========================================
@app.route('/account/export')
@login_required
@limiter.limit("10/hour")
def account_export():
    """Esporta in JSON le predizioni dell'utente (senza immagini)."""
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    docs=list(predictions_collection.find({'user_id': current_user.id}).sort('timestamp',-1))
    out=[]
    for d in docs:
        out.append({
            'id': str(d.get('_id')),
            'class': d.get('class'),
            'prob': float(d.get('prob') or 0.0),
            'confidence': d.get('confidence'),
            'feedback': d.get('feedback'),
            'correct_class_feedback': d.get('correct_class_feedback'),
            'timestamp': (d.get('timestamp') or datetime.datetime.utcnow()).isoformat(),
            'consent_training': bool(d.get('consent_training', False)),
            'image_present': bool(d.get('image_filename'))
        })
    return jsonify({'count': len(out), 'predictions': out})

# === Predict & Feedback (fetch-friendly: CSRF esenti) ========================
@app.route('/predict', methods=['POST'])
@login_required
@csrf.exempt
@limiter.limit("60/minute")
def predict():
    """
    Riceve un'immagine, effettua inferenza e salva la predizione come 'pending'.
    Ritorna json: {class, confidence, prediction_id}
    Perché csrf.exempt? Lo chiamiamo via fetch FormData; usiamo comunque sessione e login richiesto.
    """
    if not DB_READY:
        return jsonify({'error':'Database offline.'}), 503

    # Validazione presenza file
    if 'file' not in request.files or request.files['file'].filename=='':
        return jsonify({'error':'Nessun file inviato.'}), 400

    f = request.files['file']
    if f.mimetype not in ALLOWED_MIME:
        # 415 se HEIC senza supporto lato server
        msg = 'Formato non supportato. Usa JPG/PNG/WebP.'
        if ('heic' in (f.mimetype or '') or 'heif' in (f.mimetype or '')) and not _HAS_HEIF:
            msg = 'Formato HEIC/HEIF non supportato su questo server.'
        return jsonify({'error': msg}), 415

    # Costruisco nome file finale (univoco + estensione corretta)
    base = secure_filename(f.filename.rsplit('.',1)[0]) or 'img'
    ext  = MIME_TO_EXT.get(f.mimetype,'.jpg')
    uniq = f"{uuid.uuid4()}_{base}{ext}"
    final_path = Path(app.config['UPLOAD_FOLDER'])/uniq

    try:
        # Se HEIC/HEIF, decodifica e ricodifica in JPEG
        if f.mimetype in {'image/heic','image/heif'}:
            tmpdir = Path(tempfile.mkdtemp())
            tmp_in = tmpdir / f"up{ext}"
            f.save(tmp_in)
            img = Image.open(tmp_in)
            try: img = ImageOps.exif_transpose(img)
            except Exception: pass
            w, h = img.size
            if w < MIN_IMG_W or h < MIN_IMG_H:
                try: tmp_in.unlink(missing_ok=True); tmpdir.rmdir()
                except Exception: pass
                return jsonify({'error': f'Immagine troppo piccola (min {MIN_IMG_W}x{MIN_IMG_H}px).'}), 400
            img.convert('RGB').save(final_path, quality=90)
            try: tmp_in.unlink(missing_ok=True); tmpdir.rmdir()
            except Exception: pass
        else:
            # JPEG/PNG/WebP: salvo, verifico integrità e correggo orientamento
            f.save(final_path)
            Image.open(final_path).verify()  # verifica che non sia corrotto
            img = Image.open(final_path)
            try: img = ImageOps.exif_transpose(img)
            except Exception: pass
            w, h = img.size
            if w < MIN_IMG_W or h < MIN_IMG_H:
                try: final_path.unlink()
                except Exception: pass
                return jsonify({'error': f'Immagine troppo piccola (min {MIN_IMG_W}x{MIN_IMG_H}px).'}), 400

        # Conversione RGB per il modello (alcuni formati hanno alpha)
        try: img = img.convert('RGB')
        except Exception: pass

    except (UnidentifiedImageError, OSError):
        if final_path.exists():
            try: final_path.unlink()
            except Exception: pass
        return jsonify({'error':'File non valido o danneggiato.'}), 400
    except Exception as e:
        if final_path.exists():
            try: final_path.unlink()
            except Exception: pass
        return jsonify({'error': f'Errore in apertura immagine: {str(e)}'}), 400

    # Consenso per singolo upload (se assente, usa il default dell'utente)
    flag=request.form.get('consent_training')
    per_upload_consent = (flag in ('on','true','1')) if flag is not None else bool(getattr(current_user,'consent_training_default',False))

    # Inferenza PyTorch
    x=transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out=model(x)
        if TEMPERATURE!=1.0: out=out/TEMPERATURE  # calibrazione confidenza
        probs=torch.softmax(out[0],dim=0)
        p,idx=torch.max(probs,0)
        cls=CLASS_NAMES[idx.item()]
    prob=float(p.item())

    # Salva doc nel DB e marca la predizione come "pending" nella sessione
    doc={'class':cls,'confidence':f"{prob*100:.2f}%",'prob':prob,'timestamp':datetime.datetime.utcnow(),
         'image_filename':final_path.name,'user_id':current_user.id,'feedback':None,'correct_class_feedback':None,
         'consent_training':per_upload_consent,'image_deleted':False}
    inserted_id=predictions_collection.insert_one(doc).inserted_id
    session['pending_prediction_id']=str(inserted_id)
    return jsonify({'class':cls,'confidence':f"{prob*100:.2f}%",'prediction_id':str(inserted_id)})

@app.route('/feedback', methods=['POST'])
@login_required
@csrf.exempt
@limiter.limit("30/minute")
def submit_feedback():
    """
    Salva il feedback sulla predizione:
      - 'correct'
      - 'incorrect' + 'correct_class' (una delle CLASS_NAMES)
    Se non c'è consenso, elimina subito il file immagine dal disco.
    """
    if not DB_READY: return jsonify({'error':'Database offline.'}), 503
    data=request.get_json(silent=True) or {}
    pid=data.get('prediction_id'); ftype=data.get('feedback_type'); ccls=data.get('correct_class')
    if not pid or not ftype: return jsonify({'error':'Dati mancanti.'}), 400

    upd={'feedback': ftype,'feedback_timestamp': datetime.datetime.utcnow()}
    if ftype=='incorrect':
        if ccls not in CLASS_NAMES: return jsonify({'error':'Classe corretta non valida.'}), 400
        upd['correct_class_feedback']=ccls

    res=predictions_collection.update_one({'_id': ObjectId(pid), 'user_id': current_user.id},{'$set': upd})
    if res.matched_count==0: return jsonify({'error':'Predizione non trovata.'}), 404

    # no-consent: rimuovi file per evitare uso non autorizzato
    doc=predictions_collection.find_one({'_id': ObjectId(pid), 'user_id': current_user.id})
    if doc and not doc.get('consent_training', False):
        fn=doc.get('image_filename')
        if fn:
            fp=Path(app.config['UPLOAD_FOLDER'])/fn
            if fp.exists():
                try: fp.unlink()
                except Exception: pass
        predictions_collection.update_one({'_id': ObjectId(pid)},{'$set':{'image_filename':None,'image_deleted':True}})

    if session.get('pending_prediction_id')==pid: session.pop('pending_prediction_id',None)
    return jsonify({'message':'Feedback salvato.'}), 200

# === Immagini protette / delete predizione ====================================
@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    """Serve immagini solo se appartengono all'utente corrente (autorizzazione)."""
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    pred=predictions_collection.find_one({'image_filename': filename, 'user_id': current_user.id})
    if not pred: abort(403)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/prediction/<pid>/delete', methods=['POST'])
@login_required
@limiter.limit("30/minute")
def delete_prediction(pid):
    """Elimina una predizione e l'eventuale file immagine dal disco."""
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    try: oid=ObjectId(pid)
    except Exception:
        return render_template("error.html", code=400, message="ID non valido"), 400
    doc=predictions_collection.find_one({'_id': oid, 'user_id': current_user.id})
    if not doc:
        return render_template("error.html", code=404, message="Predizione non trovata"), 404
    fn=doc.get('image_filename')
    if fn:
        fp=Path(app.config['UPLOAD_FOLDER'])/fn
        if fp.exists():
            try: fp.unlink()
            except Exception: pass
    predictions_collection.delete_one({'_id': oid})
    flash('Immagine e predizione eliminate.', 'success')
    return redirect(url_for('dashboard'))

# === Admin: Export dataset ====================================================
@app.route('/export', methods=['GET'])
@login_required
def export_page():
    """Pagina admin per configurare l'export del dataset."""
    if getattr(current_user,'role','user')!='admin': abort(403)
    return render_template('export.html', class_names=CLASS_NAMES)

@app.route('/export', methods=['POST'])
@login_required
def do_export():
    """
    Genera uno ZIP con:
      - dataset.csv (righe = esempi, pesati per fiducia)
      - cartella images/ con sottocartelle per classe
    Filtri: conf minima, trust minima, date, consenso, ecc.
    """
    if getattr(current_user,'role','user')!='admin': abort(403)

    include_unverified = request.form.get('include_unverified')=='on'
    include_user_hash  = request.form.get('include_user_hash')=='on'
    scope_all          = request.form.get('scope')=='all'
    only_consented     = (request.form.get('only_consented','on')=='on')

    try: min_conf = float(request.form.get('min_conf','0') or 0)
    except Exception: min_conf = 0.0
    try: min_trust = int(float(request.form.get('min_trust','0') or 0))
    except Exception: min_trust = 0

    df = (request.form.get('date_from') or '').strip()
    dt = (request.form.get('date_to') or '').strip()

    # filtro base
    q = {}
    if not scope_all: q['user_id'] = current_user.id
    if only_consented: q['consent_training'] = True
    if df:
        try: q.setdefault('timestamp',{})['$gte']=datetime.datetime.fromisoformat(df)
        except Exception: pass
    if dt:
        try: q.setdefault('timestamp',{})['$lte']=datetime.datetime.fromisoformat(dt)
        except Exception: pass

    preds = list(predictions_collection.find(q).sort('timestamp', -1))

    # calcolo trust per gli user presenti nel set (evita ricalcoli per riga)
    uids = sorted({p.get('user_id') for p in preds if p.get('user_id')})
    trust_map = {uid: compute_user_trust(uid) for uid in uids}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        images_root = tmp / "images"
        images_root.mkdir(parents=True, exist_ok=True)

        rows = []
        for d in preds:
            # filtri dinamici
            if d.get('prob') is not None and float(d.get('prob')) < min_conf:
                continue
            uid = d.get('user_id')
            user_trust = trust_map.get(uid, 0)
            if user_trust < min_trust:
                continue

            label_final, src, base_w = compute_label_and_weight(d)
            if (src == 'unverified') and not include_unverified:
                continue

            # copia immagine (se presente)
            lf = materialize_image_local(d, images_root)
            if not lf or not lf.exists():
                # senza immagine ha poco senso esportarla
                continue

            # sotto-cartella per classe (o 'unverified')
            sub = label_final if label_final in CLASS_NAMES else 'unverified'
            td = images_root / sub
            td.mkdir(exist_ok=True, parents=True)
            final = td / lf.name
            shutil.move(str(lf), str(final))

            # peso finale: base pesato per la fiducia utente (0.5–1.0)
            trust_factor = 0.5 + 0.5 * (user_trust / 100.0)
            weight_final = base_w * trust_factor

            row = {
                'id': str(d.get('_id')),
                'filename': final.name,
                'label_final': label_final,
                'label_source': src,
                'weight': f"{weight_final:.4f}",
                'model_confidence': f"{float(d.get('prob') or 0.0):.6f}",
                'feedback': d.get('feedback') or '',
                'correct_class_feedback': d.get('correct_class_feedback') or '',
                'timestamp': (d.get('timestamp') or datetime.datetime.utcnow()).isoformat(),
                'user_trust': user_trust,
            }
            if include_user_hash:
                pepper = os.environ.get('USER_HASH_PEPPER','pepper-default-change-me')
                row['user_hash'] = hmac.new(
                    pepper.encode('utf-8'),
                    str(uid).encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()

            rows.append(row)

        # CSV
        csv_path = tmp / "dataset.csv"
        with open(csv_path,'w',newline='',encoding='utf-8') as f:
            fields = [
                'id','filename','label_final','label_source','weight',
                'model_confidence','feedback','correct_class_feedback','timestamp',
                'user_trust'
            ]
            if include_user_hash: fields.append('user_hash')
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

        # ZIP (dataset.csv + images/)
        zip_name = f"smart_waste_export_{datetime.datetime.utcnow():%Y%m%d_%H%M%S}.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            z.write(csv_path, arcname="dataset.csv")
            for p in images_root.rglob('*'):
                if p.is_file(): z.write(p, arcname=str(p.relative_to(tmp)))
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name=zip_name, mimetype='application/zip')

# === Admin: pannello fiducia (tabella + JSON) =================================
@app.route('/admin/trust')
@login_required
def admin_trust():
    """Tabella riassuntiva (admin) con metriche di fiducia utente."""
    if getattr(current_user,'role','user')!='admin': abort(403)

    # Prendiamo tutti gli utenti “attivi” (almeno una predizione)
    pipeline = [
        {"$group": {"_id": "$user_id", "total": {"$sum": 1}}},
        {"$sort": {"total": -1}},
        {"$limit": 10000}
    ]
    agg = list(predictions_collection.aggregate(pipeline))
    rows = []
    for a in agg:
        uid = a["_id"]
        total = a["total"]
        # recupero info utente
        udoc = users_collection.find_one({"_id": ObjectId(uid)}) if uid else None
        username = (udoc or {}).get("username") or uid
        email    = (udoc or {}).get("email")
        n_correct   = predictions_collection.count_documents({'user_id': uid, 'feedback':'correct'})
        n_incorrect = predictions_collection.count_documents({'user_id': uid, 'feedback':'incorrect'})
        verified = n_correct + n_incorrect
        correct_lb   = wilson_lower_bound(n_correct, verified) if verified else 0.0
        consent_rate = predictions_collection.count_documents({'user_id': uid, 'consent_training': True}) / total if total else 0.0
        verify_rate  = verified / total if total else 0.0
        trust = compute_user_trust(uid)
        rows.append(type('Row', (), {
            "username": username,
            "email": email,
            "total": total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "correct_rate_lb": correct_lb,
            "consent_rate": consent_rate,
            "trust": trust
        }))
    return render_template('admin_trust.html', rows=rows)

@app.route('/admin/trust.json')
@login_required
def admin_trust_json():
    """API JSON (admin) con metriche di fiducia utente."""
    if getattr(current_user,'role','user')!='admin': abort(403)

    pipeline = [
        {"$group": {"_id": "$user_id", "total": {"$sum": 1}}},
        {"$sort": {"total": -1}},
        {"$limit": 100000}
    ]
    agg = list(predictions_collection.aggregate(pipeline))
    out = []
    for a in agg:
        uid = a["_id"]
        total = a["total"]
        n_correct   = predictions_collection.count_documents({'user_id': uid, 'feedback':'correct'})
        n_incorrect = predictions_collection.count_documents({'user_id': uid, 'feedback':'incorrect'})
        verified = n_correct + n_incorrect
        correct_lb   = wilson_lower_bound(n_correct, verified) if verified else 0.0
        consent_rate = predictions_collection.count_documents({'user_id': uid, 'consent_training': True}) / total if total else 0.0
        verify_rate  = verified / total if total else 0.0
        trust = compute_user_trust(uid)
        out.append({
            "user_id": uid,
            "total": total,
            "verified": verified,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "correct_lb": correct_lb,
            "verify_rate": verify_rate,
            "consent_rate": consent_rate,
            "trust": trust
        })
    return jsonify({"count": len(out), "users": out})

# === Avvio (stampa URL utili e serve multi-device) ============================
def _lan_ips():
    """
    Raccoglie IP locali IPv4 non-loopback.
    Utile per stampare gli indirizzi a cui connettersi da telefono/altri PC.
    """
    ips = set()
    # IP “primario” usato per uscire su internet (trucco del socket UDP)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    # IP risolti dall’hostname
    try:
        for fam, _, _, _, sa in socket.getaddrinfo(socket.gethostname(), None):
            if fam == socket.AF_INET:
                ip = sa[0]
                if not ip.startswith("127."):
                    ips.add(ip)
    except Exception:
        pass
    return sorted(ips)

if __name__ == '__main__':
    # Default: ascolta su tutta la LAN (comodo per usare anche da smartphone/altro PC).
    # Cambia DEFAULT_HOST in '127.0.0.1' per limitarlo al solo PC locale.
    DEFAULT_HOST = '0.0.0.0'
    host = os.environ.get('FLASK_RUN_HOST', DEFAULT_HOST)
    port = int(os.environ.get('FLASK_RUN_PORT', '5000') or 5000)

    def _print_urls_once():
        """Stampa gli URL utili (localhost e LAN) una sola volta all'avvio."""
        print("")
        print(f"PC (localhost):   http://127.0.0.1:{port}/app")
        if host in ('0.0.0.0', '::'):
            ips = _lan_ips()
            for ip in ips:
                print(f"LAN (altri device): http://{ip}:{port}/app")
            if ips:
                print("\nSe il firewall chiede, consenti sulle 'Reti Private'.")
        print("\nLascia questa finestra aperta. Chiudi con CTRL+C.\n")

    # Evita doppia stampa con il reloader di Werkzeug quando debug=True
    # (Werkzeug riavvia il processo per abilitare l'auto-reload del codice)
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        _print_urls_once()

    # threaded=True per gestire più client contemporaneamente in sviluppo
    app.run(debug=True, host=host, port=port, threaded=True)
