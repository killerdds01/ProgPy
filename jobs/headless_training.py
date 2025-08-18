#!/usr/bin/env python3
"""
Headless training per Smart Waste
=================================

Questo script allena (o ri-allena) il classificatore di rifiuti **senza UI**,
a partire dallo **zip di export** generato dall’app Flask (menu Admin → Export).

Formato atteso dello zip:
- dataset.csv                 (informativo; qui NON lo usiamo)
- images/
    plastica/*.jpg|png|webp
    carta/*.*
    vetro/*.*
    organico/*.*
    indifferenziato/*.*

Pipeline in breve:
1) Scompatta lo zip in una cartella di lavoro temporanea.
2) Scansiona `images/<classe>/` e costruisce la lista campioni (path, label).
3) Suddivide in train/val/test (80/10/10).
4) Costruisce i DataLoader con augmentazioni e **WeightedRandomSampler**
   per bilanciare le classi sbilanciate.
5) **Stage-1**: carica ResNet18 pre-addestrata (ImageNet) e allena SOLO il layer
   fully-connected (testa).
6) Salva i pesi come `best_model_finetuned_light.pth`.
7) **Stage-2**: sblocca `layer4` + `fc` e rifinisce per poche epoche.
8) Salva `best_model_stage2.pth` e apprende la **temperature scaling** su validation,
   scrivendo `calibration.json`.
9) Impacchetta gli artefatti in `smart_waste_model_<timestamp>.zip`.

Uso:
    python jobs/headless_training.py <export_zip>
    # Opzionale: aggiungere SOLO campioni di "organico" da una cartella esterna
    python jobs/headless_training.py <export_zip> --only-organic-kaggle-dir <dir>

Note:
- Funziona sia su CPU sia su GPU (se PyTorch vede CUDA).
- Non rimuovere/alterare le classi in CLASS_NAMES: devono coincidere con l’app.
"""

# ==== Import standard =========================================================
import os, sys, zipfile, json, glob, random, shutil, time, copy
from datetime import datetime
from collections import Counter
import numpy as np

# ==== PyTorch / Torchvision ===================================================
import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import lr_scheduler
from PIL import Image  # (usata implicitamente dal loader di torchvision)

# ==== Config base =============================================================
# Le 5 classi devono coincidere 1:1 con l'app Flask.
CLASS_NAMES = ['plastica', 'carta', 'vetro', 'organico', 'indifferenziato']
NUM_CLASSES = len(CLASS_NAMES)

# Seed per riproducibilità di split e inizializzazioni pseudo-casuali
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Se è disponibile la GPU, PyTorch userà CUDA; altrimenti CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(s: str) -> None:
    """Stampa a schermo forzando il flush (utile nei log su servizi)."""
    print(s, flush=True)

# -----------------------------------------------------------------------------
# Utilità I/O
# -----------------------------------------------------------------------------
def unzip_export(zip_path: str, out_dir: str) -> None:
    """Estrae lo zip di export in `out_dir` (creandola se non esiste)."""
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)

def build_samples_from_export(root: str):
    """
    Legge le immagini da `root/images/<classe>/*` e costruisce la lista
    di campioni [(path_assoluto, indice_classe), ...].

    Non usiamo `dataset.csv` in questa fase: la struttura a cartelle è sufficiente.
    """
    images_root = os.path.join(root, 'images')
    if not os.path.isdir(images_root):
        raise RuntimeError('Export non valido: manca la cartella images/')

    samples = []  # lista di tuple (path, label_idx)
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    label_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}

    # Scansiona tutte le classi dichiarate in CLASS_NAMES
    for c in CLASS_NAMES:
        cdir = os.path.join(images_root, c)
        if not os.path.isdir(cdir):
            continue
        for fp in glob.glob(os.path.join(cdir, '*')):
            if os.path.splitext(fp)[1].lower() in IMG_EXTS:
                samples.append((fp, label_to_idx[c]))
    return samples

# -----------------------------------------------------------------------------
# Dataset e split
# -----------------------------------------------------------------------------
class PathDataset(Dataset):
    """
    Dataset minimale che carica le immagini dai path.
    La trasformazione (augmentazione/normalizzazione) viene passata dal chiamante.
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.classes = CLASS_NAMES
        self.class_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}

    def __len__(self):  # numero di campioni totali
        return len(self.samples)

    def __getitem__(self, idx):
        # Carica immagine da path e restituisce (tensor, label_idx)
        p, y = self.samples[idx]
        img = datasets.folder.default_loader(p)  # usa PIL internamente
        if self.transform:
            img = self.transform(img)
        return img, y

def split_samples(samples, tr=0.8, va=0.1, te=0.1):
    """
    Suddivide la lista `samples` in tre parti: train/val/test.
    Le percentuali devono sommare a 1.0.
    """
    idx = list(range(len(samples)))
    random.shuffle(idx)  # shuffle deterministico grazie al SEED
    n = len(idx)
    ntr = int(n * tr); nva = int(n * va); nte = n - ntr - nva
    s_tr = [samples[i] for i in idx[:ntr]]
    s_va = [samples[i] for i in idx[ntr:ntr + nva]]
    s_te = [samples[i] for i in idx[ntr + nva:]]
    return s_tr, s_va, s_te

def build_loaders(train_list, val_list, test_list, batch=64):
    """
    Crea i DataLoader per train/val/test.
    - Train: augmentazioni leggere + normalizzazione ImageNet.
    - Val/Test: resize+centercrop + normalizzazione.
    - **WeightedRandomSampler** sul train per bilanciare dataset sbilanciati.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256)),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }

    ds_tr = PathDataset(train_list, data_transforms['train'])
    ds_va = PathDataset(val_list,   data_transforms['test'])
    ds_te = PathDataset(test_list,  data_transforms['test'])

    # Calcola i pesi per classe -> pesi per campione (inverse frequency)
    counts = Counter([y for _, y in train_list])
    class_count  = np.array([counts.get(c, 1) for c in range(NUM_CLASSES)], dtype=np.float32)
    class_weight = 1.0 / class_count
    sample_weight = np.array([class_weight[y] for _, y in train_list], dtype=np.float32)

    sampler = WeightedRandomSampler(
        weights=sample_weight,
        num_samples=len(sample_weight),  # stessa size del dataset di train
        replacement=True                 # con rimpiazzamento per bilanciare
    )

    # num_workers=2: parallelizza il caricamento; alza se hai SSD+RAM
    tr_loader = DataLoader(ds_tr, batch_size=batch, sampler=sampler, num_workers=2)
    va_loader = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=2)
    te_loader = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=2)
    return ds_tr, ds_va, ds_te, tr_loader, va_loader, te_loader

# -----------------------------------------------------------------------------
# Training loop (con early stopping su validation loss)
# -----------------------------------------------------------------------------
def train_model(model, criterion, optimizer, scheduler, loaders, num_epochs=30, patience=8, tag=''):
    """
    Allena `model` usando:
    - `criterion` (CrossEntropy con label smoothing)
    - `optimizer` (Adam)
    - `scheduler` (ReduceLROnPlateau su val_loss)
    - early stopping se nessun miglioramento per `patience` epoche

    Ritorna il modello con i **migliori pesi trovati su validation**.
    """
    train_loader, val_loader = loaders
    best_w = copy.deepcopy(model.state_dict())
    best_loss = 1e9; best_acc = 0.0
    epochs_no_improve = 0

    for ep in range(num_epochs):
        log(f"[{tag}] Epoch {ep+1}/{num_epochs}")

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            loader = train_loader if phase == 'train' else val_loader

            run_loss = 0.0; corr = 0; n = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == 'train'):
                    out = model(x)
                    loss = criterion(out, y)
                    _, pred = torch.max(out, 1)  # predizione top-1

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # evita gradient explosion
                        optimizer.step()

                run_loss += loss.item() * x.size(0)
                corr     += torch.sum(pred == y)
                n        += x.size(0)

            ep_loss = run_loss / max(1, n)
            ep_acc  = (corr.double() / max(1, n)).item()

            if phase == 'val':
                scheduler.step(ep_loss)  # ReduceLROnPlateau osserva la val_loss

                # Migliora se perdita diminuisce (o a parità di loss, acc aumenta)
                improved = (ep_loss < best_loss - 1e-6) or (abs(ep_loss - best_loss) < 1e-6 and ep_acc > best_acc)
                if improved:
                    best_loss = ep_loss
                    best_acc  = ep_acc
                    best_w    = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        log(f"val_loss={best_loss:.4f} best_acc={best_acc:.4f} no_improve={epochs_no_improve}/{patience}")
        if epochs_no_improve >= patience:
            log("Early stopping.")
            break

    # Ripristina i migliori pesi trovati su validation
    model.load_state_dict(best_w)
    return model

# -----------------------------------------------------------------------------
# Temperature scaling (post-hoc calibration)
# -----------------------------------------------------------------------------
def learn_temperature(model, val_loader, device, grid=np.linspace(0.5, 3.0, 26)):
    """
    Stima la temperatura T (softmax(logits/T)) che minimizza la NLL
    sul validation set. Semplice grid search su [0.5, 3.0].
    """
    import torch.nn.functional as F

    model.eval()
    logits_list = []; labels_list = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits_list.append(logits.cpu()); labels_list.append(y.cpu())

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    best_T, best_nll = 1.0, 1e9
    for T in grid:
        nll = F.cross_entropy(logits / T, labels).item()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Uso: headless_training.py <export_zip> [--only-organic-kaggle-dir <dir>]")
        sys.exit(1)

    export_zip = sys.argv[1]
    only_org_dir = None
    if '--only-organic-kaggle-dir' in sys.argv:
        i = sys.argv.index('--only-organic-kaggle-dir')
        only_org_dir = sys.argv[i + 1] if i + 1 < len(sys.argv) else None

    # Cartella di lavoro temporanea basata sul timestamp
    work = f"wrk_{int(time.time())}"
    os.makedirs(work, exist_ok=True)

    # 1) Scompatta e costruisci i campioni
    unzip_export(export_zip, work)
    samples = build_samples_from_export(work)

    # 1bis) Opzionale: aggiungi SOLO campioni "organico" da una cartella esterna (es. Kaggle)
    if only_org_dir and os.path.isdir(only_org_dir):
        add = []
        for fp in glob.glob(os.path.join(only_org_dir, "*")):
            if os.path.splitext(fp)[1].lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
                add.append((fp, CLASS_NAMES.index('organico')))
        samples.extend(add)
        print(f"Aggiunti da Kaggle (organico): {len(add)}")

    # Safety check: serve un minimo di dati
    if len(samples) < 50:
        raise RuntimeError("Pochi campioni dopo export; servono più dati con feedback.")

    # 2) Split e DataLoader
    tr, va, te = split_samples(samples, 0.8, 0.1, 0.1)
    ds_tr, ds_va, ds_te, tr_loader, va_loader, te_loader = build_loaders(tr, va, te, batch=64)

    # 3) Stage-1: allena solo la testa (fc) della ResNet18 pre-addestrata
    import torchvision
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # Congela tutto tranne la testa
    for p in model.parameters():   p.requires_grad = False
    for p in model.fc.parameters(): p.requires_grad = True
    model = model.to(device)

    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    sch  = lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=7, factor=0.1, min_lr=1e-6)

    model = train_model(model, crit, opt, sch, (tr_loader, va_loader),
                        num_epochs=60, patience=10, tag='Stage-1')

    # Salva il risultato dello stage-1 (usato anche dall’app)
    torch.save(model.state_dict(), 'best_model_finetuned_light.pth')

    # 4) Stage-2: sblocca layer4 + fc per una rifinitura leggera
    model2 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model2.fc = nn.Linear(model2.fc.in_features, NUM_CLASSES)
    model2.load_state_dict(torch.load('best_model_finetuned_light.pth', map_location=device))
    model2.to(device)

    for p in model2.parameters():           p.requires_grad = False
    for p in model2.layer4.parameters():    p.requires_grad = True
    for p in model2.fc.parameters():        p.requires_grad = True

    # BatchNorm in eval per stabilità quando si sblocca una parte della rete
    for m in model2.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    opt2 = optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=3e-4, weight_decay=1e-4)
    sch2 = lr_scheduler.ReduceLROnPlateau(opt2, mode='min', patience=5, factor=0.2, min_lr=1e-6)

    model2 = train_model(model2, crit, opt2, sch2, (tr_loader, va_loader),
                         num_epochs=15, patience=5, tag='Stage-2')

    torch.save(model2.state_dict(), 'best_model_stage2.pth')

    # 5) Calibrazione della confidenza (temperature scaling) su validation
    T = learn_temperature(model2, va_loader, device)
    with open('calibration.json', 'w') as f:
        json.dump({'temperature': float(T)}, f)

    # 6) Packaging degli artefatti per l’app/backup
    ver = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"smart_waste_model_{ver}"
    os.makedirs(outdir, exist_ok=True)

    for f in ['best_model_finetuned_light.pth', 'best_model_stage2.pth', 'calibration.json']:
        if os.path.exists(f):
            shutil.copy(f, os.path.join(outdir, f))

    # Mappa indice->nome classe (usata dall’app in fase di inferenza)
    with open(os.path.join(outdir, 'class_index.json'), 'w', encoding='utf-8') as f:
        json.dump({i: n for i, n in enumerate(CLASS_NAMES)}, f, ensure_ascii=False, indent=2)

    # Zip finale
    zip_name = outdir + '.zip'
    shutil.make_archive(outdir, 'zip', outdir)
    print("Artefatti salvati:", zip_name)

# Entry-point
if __name__ == '__main__':
    main()
