# === Smart Waste · Flask App (FINAL + User Trust) =============================
# Requisiti: Flask, Flask-Login, Flask-WTF, flask-cors, python-dotenv,
#            pymongo, pillow, torch, torchvision, email-validator
# Opzionale: Flask-Limiter (se assente, i limit diventano no-op)
# =============================================================================

import os, io, csv, hmac, json, uuid, time, shutil, zipfile, datetime, tempfile, subprocess, hashlib
from pathlib import Path
from typing import Optional
from datetime import timedelta

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
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

# --- MongoDB ------------------------------------------------------------------
from pymongo import MongoClient
from bson.objectid import ObjectId

# --- Env ----------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# === Flask base ================================================================
# NB: abbiamo cartella 'static/static' su disco ma la esponiamo come '/static'
app = Flask(
    __name__,
    static_folder=os.path.join('static', 'static'),
    static_url_path='/static',
    template_folder='templates'
)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'DEV_KEY_CHANGE_ME')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

app.config.update(
    REMEMBER_COOKIE_DURATION=timedelta(days=14),
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=45),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,   # True in produzione (HTTPS)
    REMEMBER_COOKIE_HTTPONLY=True,
    REMEMBER_COOKIE_SECURE=False,  # True in produzione (HTTPS)
)

# Limiter (no-op se il pacchetto non è installato)
limiter = Limiter(key_func=get_remote_address, app=app, default_limits=["200/hour"])

UPLOAD_FOLDER  = 'uploaded_images'
PROFILE_FOLDER = 'profile_photos'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(PROFILE_FOLDER).mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROFILE_FOLDER'] = PROFILE_FOLDER

ALLOWED_MIME = {'image/jpeg', 'image/png', 'image/webp'}
MIME_TO_EXT  = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp'}

CORS(app)
csrf = CSRFProtect(app)

# === Login manager =============================================================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# === MongoDB (autostart dev) ==================================================
MONGO_URL    = os.environ.get('MONGO_URL', 'mongodb://127.0.0.1:27017/')
MONGO_PORT   = int(os.environ.get('MONGO_PORT', '27017') or 27017)
MONGO_AUTO   = (os.environ.get('MONGO_AUTOSTART', 'false').lower() == 'true')
MONGO_BIN    = os.environ.get('MONGO_BIN', '').strip().strip('"').strip("'")
MONGO_DBPATH = os.environ.get('MONGO_DBPATH', './mongo_data')

client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=2000, connectTimeoutMS=2000)

def _mongo_up() -> bool:
    try:
        client.admin.command('ping')
        return True
    except Exception as e:
        print(f"[WARN] MongoDB non raggiungibile: {e}")
        return False

def _try_start_mongo():
    """Avvia mongod localmente se abilitato (solo dev)."""
    if not MONGO_AUTO or _mongo_up():
        return _mongo_up()
    mongod = MONGO_BIN or shutil.which("mongod")
    if not mongod or not Path(mongod).exists():
        print(f"[MONGO] mongod non trovato. MONGO_BIN='{os.environ.get('MONGO_BIN','')}'.")
        return False
    Path(MONGO_DBPATH).mkdir(parents=True, exist_ok=True)
    args = [mongod, "--dbpath", str(MONGO_DBPATH), "--port", str(MONGO_PORT), "--bind_ip", "127.0.0.1", "--quiet"]
    flags = 0
    if os.name == "nt":
        flags = 0x00000200 | 0x00000008  # NEW_PROCESS_GROUP | DETACHED
    try:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        time.sleep(2.0)
        if _mongo_up():
            print(f"[MONGO] mongod avviato su 127.0.0.1:{MONGO_PORT} (pid={proc.pid})")
            return True
    except Exception as e:
        print(f"[MONGO] Avvio mongod fallito: {e}")
    return False

DB_READY = _mongo_up() or (_try_start_mongo() and _mongo_up())
db  = client.smart_waste if DB_READY else None
predictions_collection = db.predictions if DB_READY else None
users_collection       = db.users if DB_READY else None

# === Modello ==================================================================
CLASS_NAMES = ['plastica', 'carta', 'vetro', 'organico', 'indifferenziato']
device = torch.device("cpu")
MODEL_PATH = os.environ.get('MODEL_PATH', 'best_model_finetuned_light.pth')

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
try:
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
except TypeError:
    state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state); model.to(device).eval()

# Temperature scaling (opzionale)
TEMPERATURE = 1.0
try:
    with open('calibration.json','r',encoding='utf-8') as f:
        TEMPERATURE = float(json.load(f).get('temperature',1.0))
    print(f"[CALIB] T={TEMPERATURE}")
except Exception:
    print("[CALIB] T=1.0")

transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# === Forms ====================================================================
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
    """Applica un 'pepper' lato server alla password prima di hasharla/verificarla."""
    return f"{os.environ.get('PASSWORD_PEPPER','')}{pw}"

class User(UserMixin):
    """Wrapper user per Flask-Login (lettura da Mongo)."""
    def __init__(self, d):
        self.id = str(d['_id'])
        self.username = d.get('username') or d.get('email')
        self.email = d.get('email')
        self.first_name = d.get('first_name')
        self.last_name  = d.get('last_name')
        self.role = d.get('role','user')
        self.consent_training_default = bool(d.get('consent_training_default', False))
        self.profile_image = d.get('profile_image')
        self.trust = float(d.get('trust', 0.5))  # opzionale: utile da mostrare

@login_manager.user_loader
def load_user(user_id):
    if not DB_READY:
        return None
    doc = users_collection.find_one({"_id": ObjectId(user_id)})
    return User(doc) if doc else None

# --- Trust system --------------------------------------------------------------
# Modello: Beta(a,b) con priori a=2,b=2 ⇒ trust iniziale = 0.5
# - feedback 'correct'    -> a += 1.0
# - feedback 'incorrect'  -> a += 0.25 (premio la correzione), b += 0.75
# Trust = a / (a + b)
def update_user_trust(user_id: str, feedback: str, has_correction: bool = False) -> float:
    try:
        u = users_collection.find_one({'_id': ObjectId(user_id)})
    except Exception:
        u = None
    if not u:
        return 0.5

    stats = u.get('trust_stats') or {'a': 2.0, 'b': 2.0}
    a = float(stats.get('a', 2.0))
    b = float(stats.get('b', 2.0))

    if feedback == 'correct':
        a += 1.0
    elif feedback == 'incorrect':
        a += 0.25 if has_correction else 0.0
        b += 0.75

    trust = a / (a + b)
    users_collection.update_one(
        {'_id': ObjectId(user_id)},
        {'$set': {'trust': trust, 'trust_stats': {'a': a, 'b': b}}}
    )
    return trust

# Migrazione soft per utenti esistenti (se manca trust/trust_stats)
if DB_READY:
    try:
        users_collection.update_many(
            {'trust': {'$exists': False}},
            {'$set': {'trust': 0.5, 'trust_stats': {'a': 2.0, 'b': 2.0}}}
        )
    except Exception:
        pass

# === Admin bootstrap (opzionale) ==============================================
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
                'role':'admin','consent_training_default':False,'profile_image':None,
                'trust': 0.5, 'trust_stats': {'a': 2.0, 'b': 2.0}
            })
            print(f"[BOOT] Creato admin: {admin_username}")
    except Exception as e:
        print(f"[WARN] Bootstrap admin saltato: {e}")

# --- Label & copy helpers ------------------------------------------------------
def compute_label_and_weight(doc):
    """
    Stabilisce label_final, source e base weight (senza fiducia).
    - feedback == 'correct'                -> predetta,        'user_verified', 1.0
    - feedback == 'incorrect' + correzione -> correzione,      'user_corrected', 0.9
    - altrimenti (non verificata)          -> predetta,        'unverified',    0.3
    """
    pred = doc.get('class','')
    fb   = (doc.get('feedback') or '').strip().lower()
    corr = (doc.get('correct_class_feedback') or '').strip().lower()
    if fb == 'correct' and pred in CLASS_NAMES:     return pred,'user_verified',1.0
    if fb == 'incorrect' and corr in CLASS_NAMES:   return corr,'user_corrected',0.9
    if pred in CLASS_NAMES:                         return pred,'unverified',0.3
    return '','unverified',0.3

def materialize_image_local(doc, root: Path) -> Optional[Path]:
    """Copia l'immagine su una cartella temporanea per lo ZIP di export."""
    fname = doc.get('image_filename')
    if not fname:
        return None
    src = Path(app.config['UPLOAD_FOLDER'])/fname
    if not src.exists():
        return None
    dst = root/fname
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src,dst)
    return dst

# === Middleware & headers =====================================================
@app.before_request
def idle_logout():
    """Logout automatico per inattività."""
    if not current_user.is_authenticated:
        return
    now = datetime.datetime.utcnow().timestamp()
    last = session.get('last_seen', now)
    if now - last > app.config['PERMANENT_SESSION_LIFETIME'].total_seconds():
        logout_user()
        session.clear()
        flash('Sessione scaduta per inattività. Accedi di nuovo.','info')
        return redirect(url_for('login'))
    session['last_seen'] = now

@app.before_request
def require_db_if_needed():
    """Consenti solo alcune pagine se il DB non è pronto."""
    if DB_READY:
        return
    if (request.endpoint or '') not in {'landing','login','register','static'}:
        return render_template("error.html", code=503, message="Database offline. Avvia MongoDB e ricarica."), 503

@app.before_request
def force_feedback_before_navigation():
    """
    Blocca la navigazione se c'è una predizione pendente non confermata,
    tranne per questi endpoint ammessi (upload, feedback, ecc.).
    """
    if not current_user.is_authenticated:
        return
    if not session.get('pending_prediction_id'):
        return
    allowed={'app','predict','submit_feedback','uploaded_file','login','register','logout','static',
             'dashboard','profile','profile_photo_upload','profile_photo_get','delete_prediction',
             'account_delete','profile_password','account_export','export_page','do_export'}
    if request.method=='GET' and (request.endpoint or '') not in allowed:
        flash('Conferma prima la predizione in corso per proseguire.','error')
        return redirect(url_for('app'))

@app.after_request
def set_security_headers(resp):
    """Header di sicurezza minimi (CSP, Referrer, ecc.)."""
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
    """Variabili globali per i template."""
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

# === Routes ===================================================================
@app.route('/')
def landing():
    pending = 0
    if current_user.is_authenticated and not session.get('pending_prediction_id') and DB_READY:
        pending = predictions_collection.count_documents({'user_id': current_user.id, 'feedback': None})
    return render_template('landing.html', pending_count=pending)

@app.route('/app', endpoint='app')
@login_required
def classify():
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

# ---- Auth --------------------------------------------------------------------
@app.route('/register', methods=['GET','POST'])
@limiter.limit("10/minute")
def register():
    form = RegistrationForm()
    if current_user.is_authenticated:
        return redirect(url_for('app'))
    if request.method=='POST' and form.validate_on_submit():
        if not DB_READY:
            flash('Database offline: impossibile registrare ora.','error')
            return render_template('register.html', form=form)
        email=form.email.data.strip().lower()
        username=(form.username.data or '').strip()
        if users_collection.find_one({'email': email}):
            flash('Email già registrata.','error')
            return render_template('register.html', form=form)
        users_collection.insert_one({
            'username': username or email.split('@')[0],
            'first_name': form.first_name.data.strip(),
            'last_name' : form.last_name.data.strip(),
            'email': email,
            'password': generate_password_hash(_peppered(form.password.data)),
            'role':'user',
            'created_at': datetime.datetime.utcnow(),
            'consent_training_default': bool(form.consent_training_default.data),
            'profile_image': None,
            # trust defaults
            'trust': 0.5,
            'trust_stats': {'a': 2.0, 'b': 2.0},
        })
        flash('Registrazione completata! Ora accedi.','success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET','POST'])
@limiter.limit("10/minute")
def login():
    form = LoginForm()
    if current_user.is_authenticated:
        return redirect(url_for('app'))
    if request.method=='POST' and form.validate_on_submit():
        if not DB_READY:
            flash('Database offline: impossibile accedere ora.','error')
            return render_template('login.html', form=form)
        key=form.username.data.strip()
        q={'$or':[{'email':key.lower()},{'username':key}]}
        u=users_collection.find_one(q)
        if u and check_password_hash(u['password'], _peppered(form.password.data)):
            login_user(User(u), remember=form.remember.data)
            session.permanent=True
            session['last_seen']=datetime.datetime.utcnow().timestamp()
            flash('Login effettuato!','success')
            return redirect(url_for('app'))
        flash('Credenziali errate.','error')
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    logout_user()
    session.clear()
    flash('Logout effettuato.','info')
    return redirect(url_for('landing'))

# ---- Dashboard + filtri ------------------------------------------------------
@app.route('/dashboard')
@login_required
def dashboard():
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

# ---- Profilo -----------------------------------------------------------------
@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    if request.method=='POST':
        first=(request.form.get('first_name') or '').strip()
        last =(request.form.get('last_name') or '').strip()
        cons = True if request.form.get('consent_training_default') in ('on','true','1') else False
        users_collection.update_one({'_id': ObjectId(current_user.id)},
                                    {'$set': {'first_name':first,'last_name':last,'consent_training_default':cons}})
        flash('Profilo aggiornato.','success')
        return redirect(url_for('profile'))
    usr=users_collection.find_one({'_id': ObjectId(current_user.id)}) if DB_READY else {}
    return render_template('profile.html', user=usr)

@app.route('/profile/photo', methods=['POST'])
@login_required
@limiter.limit("20/minute")
def profile_photo_upload():
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    f = request.files.get('photo')
    if not f or f.filename=='':
        flash('Nessun file selezionato.','error'); return redirect(url_for('profile'))
    if f.mimetype not in ALLOWED_MIME:
        flash('Formato non supportato (JPG/PNG/WebP).','error'); return redirect(url_for('profile'))
    ext=MIME_TO_EXT.get(f.mimetype,'.jpg')
    filename=f"profile_{current_user.id}{ext}"
    path=Path(app.config['PROFILE_FOLDER'])/filename
    try:
        tmp=Path(tempfile.mkdtemp())/f"tmp{ext}"
        f.save(tmp)
        img=Image.open(tmp).convert('RGB')
        img.thumbnail((512,512))
        img.save(path,quality=88)
        tmp.unlink(missing_ok=True)
        users_collection.update_one({'_id': ObjectId(current_user.id)}, {'$set': {'profile_image': filename}})
        flash('Foto profilo aggiornata.','success')
    except Exception:
        flash('Errore durante il salvataggio della foto.','error')
    return redirect(url_for('profile'))

@app.route('/profile/photo/<filename>')
@login_required
def profile_photo_get(filename):
    is_owner = filename.startswith(f"profile_{current_user.id}")
    if not is_owner and getattr(current_user,'role','user')!='admin':
        abort(403)
    return send_from_directory(app.config['PROFILE_FOLDER'], filename)

@app.route('/profile/password', methods=['POST'])
@login_required
@limiter.limit("10/hour")
def profile_password():
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    old=request.form.get('old_password') or ''
    new=request.form.get('new_password') or ''
    conf=request.form.get('confirm_password') or ''
    if len(new)<6 or new!=conf:
        flash('La nuova password deve avere almeno 6 caratteri e coincidere con la conferma.','error')
        return redirect(url_for('profile'))
    u=users_collection.find_one({'_id': ObjectId(current_user.id)})
    if not u or not check_password_hash(u['password'], _peppered(old)):
        flash('Password attuale errata.','error')
        return redirect(url_for('profile'))
    users_collection.update_one({'_id': ObjectId(current_user.id)},
                                {'$set':{'password': generate_password_hash(_peppered(new))}})
    flash('Password aggiornata correttamente.','success')
    return redirect(url_for('profile'))

# ---- Cancellazione account ---------------------------------------------------
@app.route('/account/delete', methods=['POST'])
@login_required
@limiter.limit("5/hour")
def account_delete():
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    pwd=request.form.get('password') or ''
    u=users_collection.find_one({'_id': ObjectId(current_user.id)})
    if not u or not check_password_hash(u['password'], _peppered(pwd)):
        flash('Password errata.','error'); return redirect(url_for('profile'))
    # cancella immagini/predizioni
    for p in predictions_collection.find({'user_id': current_user.id}):
        fn=p.get('image_filename')
        if fn:
            fp=Path(app.config['UPLOAD_FOLDER'])/fn
            if fp.exists():
                try: fp.unlink()
                except Exception: pass
    predictions_collection.delete_many({'user_id': current_user.id})
    # foto profilo
    if u.get('profile_image'):
        pp=Path(app.config['PROFILE_FOLDER'])/u['profile_image']
        if pp.exists():
            try: pp.unlink()
            except Exception: pass
    users_collection.delete_one({'_id': ObjectId(current_user.id)})
    logout_user(); session.clear()
    flash('Account eliminato definitivamente.','success')
    return redirect(url_for('landing'))

# ---- Esportazione dati utente (JSON) ----------------------------------------
@app.route('/account/export')
@login_required
@limiter.limit("10/hour")
def account_export():
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

# ---- Predict & Feedback ------------------------------------------------------
@app.route('/predict', methods=['POST'])
@login_required
@csrf.exempt                  # esentato se chiami via fetch senza token CSRF
@limiter.limit("60/minute")
def predict():
    if not DB_READY:
        return jsonify({'error':'Database offline.'}), 503
    if 'file' not in request.files or request.files['file'].filename=='':
        return jsonify({'error':'Nessun file inviato.'}), 400
    f=request.files['file']
    if f.mimetype not in ALLOWED_MIME:
        return jsonify({'error':'Formato non supportato. Usa JPG/PNG/WebP.'}), 400

    base=secure_filename(f.filename.rsplit('.',1)[0]) or 'img'
    ext=MIME_TO_EXT.get(f.mimetype,'.jpg')
    uniq=f"{uuid.uuid4()}_{base}{ext}"
    path=Path(app.config['UPLOAD_FOLDER'])/uniq
    try:
        f.save(path)
        Image.open(path).verify()
        img=Image.open(path).convert('RGB')
    except (UnidentifiedImageError,OSError):
        path.exists() and path.unlink()
        return jsonify({'error':'File non valido o danneggiato.'}), 400

    # consenso per l'uso della singola immagine
    flag=request.form.get('consent_training')
    per_upload_consent = (flag in ('on','true','1')) if flag is not None else bool(getattr(current_user,'consent_training_default',False))

    # inferenza
    x=transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out=model(x)
        if TEMPERATURE!=1.0:
            out=out/TEMPERATURE
        probs=torch.softmax(out[0],dim=0)
        p,idx=torch.max(probs,0)
        cls=CLASS_NAMES[idx.item()]
    prob=float(p.item())

    # salvataggio predizione
    doc={'class':cls,'confidence':f"{prob*100:.2f}%",'prob':prob,'timestamp':datetime.datetime.utcnow(),
         'image_filename':uniq,'user_id':current_user.id,'feedback':None,'correct_class_feedback':None,
         'consent_training':per_upload_consent,'image_deleted':False}
    inserted_id=predictions_collection.insert_one(doc).inserted_id
    session['pending_prediction_id']=str(inserted_id)
    return jsonify({'class':cls,'confidence':f"{prob*100:.2f}%",'prediction_id':str(inserted_id)})

@app.route('/feedback', methods=['POST'])
@login_required
@csrf.exempt                  # esentato se chiami via fetch senza token CSRF
@limiter.limit("30/minute")
def submit_feedback():
    if not DB_READY:
        return jsonify({'error':'Database offline.'}), 503
    data=request.get_json(silent=True) or {}
    pid=data.get('prediction_id'); ftype=data.get('feedback_type'); ccls=data.get('correct_class')
    if not pid or not ftype:
        return jsonify({'error':'Dati mancanti.'}), 400

    upd={'feedback': ftype,'feedback_timestamp': datetime.datetime.utcnow()}
    if ftype=='incorrect':
        if ccls not in CLASS_NAMES:
            return jsonify({'error':'Classe corretta non valida.'}), 400
        upd['correct_class_feedback']=ccls

    res=predictions_collection.update_one({'_id': ObjectId(pid), 'user_id': current_user.id},{'$set': upd})
    if res.matched_count==0:
        return jsonify({'error':'Predizione non trovata.'}), 404

    # Aggiorna fiducia utente in base al feedback
    update_user_trust(current_user.id, ftype, bool(ccls))

    # Se l'utente NON consente l'uso per training, rimuoviamo il file
    doc=predictions_collection.find_one({'_id': ObjectId(pid), 'user_id': current_user.id})
    if doc and not doc.get('consent_training', False):
        fn=doc.get('image_filename')
        if fn:
            fp=Path(app.config['UPLOAD_FOLDER'])/fn
            if fp.exists():
                try: fp.unlink()
                except Exception: pass
        predictions_collection.update_one({'_id': ObjectId(pid)},
                                          {'$set':{'image_filename':None,'image_deleted':True}})

    if session.get('pending_prediction_id')==pid:
        session.pop('pending_prediction_id',None)
    return jsonify({'message':'Feedback salvato.'}), 200

# ---- Immagini protette / delete predizione -----------------------------------
@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    pred=predictions_collection.find_one({'image_filename': filename, 'user_id': current_user.id})
    if not pred:
        abort(403)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/prediction/<pid>/delete', methods=['POST'])
@login_required
@limiter.limit("30/minute")
def delete_prediction(pid):
    if not DB_READY:
        return render_template("error.html", code=503, message="Database offline."), 503
    try:
        oid=ObjectId(pid)
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
    flash('Immagine e predizione eliminate.','success')
    return redirect(url_for('dashboard'))

# ---- Export (admin) ----------------------------------------------------------
@app.route('/export', methods=['GET'])
@login_required
def export_page():
    if getattr(current_user,'role','user')!='admin':
        abort(403)
    return render_template('export.html', class_names=CLASS_NAMES)

@app.route('/export', methods=['POST'])
@login_required
def do_export():
    if getattr(current_user,'role','user')!='admin':
        abort(403)

    include_unverified = request.form.get('include_unverified')=='on'
    include_user_hash  = request.form.get('include_user_hash')=='on'
    scope_all          = request.form.get('scope')=='all'
    only_consented     = (request.form.get('only_consented','on')=='on')

    # conf. minima (0..1)
    try:
        min_conf=float(request.form.get('min_conf','0') or 0)
    except Exception:
        min_conf=0.0

    # fiducia minima utente (0..1)
    try:
        min_trust=float(request.form.get('min_trust','0') or 0)
    except Exception:
        min_trust=0.0

    df=(request.form.get('date_from') or '').strip()
    dt=(request.form.get('date_to') or '').strip()

    q={}
    if not scope_all:
        q['user_id']=current_user.id
    if only_consented:
        q['consent_training']=True
    if df:
        try: q.setdefault('timestamp',{})['$gte']=datetime.datetime.fromisoformat(df)
        except Exception: pass
    if dt:
        try: q.setdefault('timestamp',{})['$lte']=datetime.datetime.fromisoformat(dt)
        except Exception: pass

    # helper: recupera fiducia dell'utente proprietario della predizione
    def _user_trust_from_pred(pred) -> float:
        uid = pred.get('user_id')
        if not uid:
            return 0.5
        try:
            u = users_collection.find_one({'_id': ObjectId(uid)})
            if u and isinstance(u.get('trust'), (int, float)):
                return float(u['trust'])
        except Exception:
            pass
        return 0.5

    preds=list(predictions_collection.find(q).sort('timestamp',-1))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp=Path(tmpdir)
        images_root=tmp/"images"
        images_root.mkdir(parents=True, exist_ok=True)
        rows=[]; copied=0; skipped=0

        for d in preds:
            # confidenza minima del modello
            if d.get('prob') is not None and float(d.get('prob')) < min_conf:
                continue

            # filtro fiducia utente
            user_trust = _user_trust_from_pred(d)
            if user_trust < min_trust:
                continue

            # etichetta e peso base (prima della fiducia)
            label_final, src, base_w = compute_label_and_weight(d)
            if (src=='unverified') and not include_unverified:
                continue

            # copia immagine (se presente/consentita)
            lf=materialize_image_local(d, images_root)
            if not lf or not lf.exists():
                skipped+=1; continue

            sub=label_final if label_final in CLASS_NAMES else 'unverified'
            td=images_root/sub; td.mkdir(exist_ok=True, parents=True)
            final=td/lf.name
            shutil.move(str(lf),str(final))
            copied+=1

            # peso finale = base * (0.5 + 0.5*trust)  -> range sempre 0..1
            trust_factor = 0.5 + 0.5 * user_trust
            final_weight = max(0.0, min(1.0, base_w * trust_factor))

            row={
                'id':str(d.get('_id')),
                'filename':final.name,
                'label_final':label_final,
                'label_source':src,
                'weight':f"{final_weight:.4f}",
                'model_confidence':f"{float(d.get('prob') or 0.0):.6f}",
                'feedback':d.get('feedback') or '',
                'correct_class_feedback': d.get('correct_class_feedback') or '',
                'timestamp': (d.get('timestamp') or datetime.datetime.utcnow()).isoformat(),
                'user_trust': f"{user_trust:.4f}",
            }
            if include_user_hash:
                pepper=os.environ.get('USER_HASH_PEPPER','pepper-default-change-me')
                row['user_hash']=hmac.new(
                    pepper.encode('utf-8'),
                    str(d.get('user_id')).encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()

            rows.append(row)

        # CSV
        csv_path=tmp/"dataset.csv"
        with open(csv_path,'w',newline='',encoding='utf-8') as f:
            fields=['id','filename','label_final','label_source','weight',
                    'model_confidence','feedback','correct_class_feedback',
                    'timestamp','user_trust']
            if include_user_hash:
                fields.append('user_hash')
            w=csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

        # ZIP
        zip_name=f"smart_waste_export_{datetime.datetime.utcnow():%Y%m%d_%H%M%S}.zip"
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            z.write(csv_path, arcname="dataset.csv")
            for p in images_root.rglob('*'):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(tmp)))
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name=zip_name, mimetype='application/zip')

# === Run ======================================================================
if __name__ == '__main__':
    app.run(debug=True, host=os.environ.get('FLASK_RUN_HOST','127.0.0.1'),
            port=int(os.environ.get('FLASK_RUN_PORT','5000') or 5000))
