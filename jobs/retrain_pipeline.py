#!/usr/bin/env python3
import os, sys, time, requests, subprocess, tempfile, json

APP_BASE = os.environ.get('APP_BASE', 'http://127.0.0.1:5000')
ADMIN_API_TOKEN = os.environ.get('ADMIN_API_TOKEN')

def api_export(save_path, include_unverified=False):
    url = f"{APP_BASE}/api/export?scope=all&include_unverified={'true' if include_unverified else 'false'}"
    r = requests.post(url, headers={'X-Admin-Token': ADMIN_API_TOKEN}, timeout=600)
    r.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(r.content)
    return save_path

def run_training(export_zip, only_organic_dir=None):
    cmd = ['python', 'jobs/headless_training.py', export_zip]
    if only_organic_dir:
        cmd += ['--only-organic-kaggle-dir', only_organic_dir]
    print("Eseguo:", ' '.join(cmd))
    subprocess.check_call(cmd)
    # trova zip
    zips = [f for f in os.listdir('.') if f.startswith('smart_waste_model_') and f.endswith('.zip')]
    zips.sort()
    return zips[-1] if zips else None

def api_reload(model_path='best_model_finetuned_light.pth', calib_path='calibration.json'):
    url = f"{APP_BASE}/api/reload-model"
    r = requests.post(url, headers={'X-Admin-Token': ADMIN_API_TOKEN},
                      json={'model_path': model_path, 'calib_path': calib_path}, timeout=120)
    r.raise_for_status()
    return r.json()

def main():
    if not ADMIN_API_TOKEN:
        print("Errore: ADMIN_API_TOKEN non impostato.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as td:
        export_zip = os.path.join(td, 'export.zip')
        print("Richiedo export...")
        api_export(export_zip, include_unverified=False)

        print("Avvio training headless...")
        # opzionale: passare only_organic_dir per bilanciare "organico"
        artifacts_zip = run_training(export_zip, only_organic_dir=None)
        print("Artefatti:", artifacts_zip)

        print("Ricarico modello nell'app...")
        res = api_reload('best_model_finetuned_light.pth', 'calibration.json')
        print("Reload:", res)

if __name__ == '__main__':
    main()
