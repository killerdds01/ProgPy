(function () {
  console.log('[index.js] loaded');

  // ------- CSRF + preload ----------------------------------------------------
  let CSRF = '';
  try {
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta?.content) CSRF = meta.content;
  } catch (_) {}

  let PRELOAD = null;
  let PRELOAD_IMG_URL = '';
  try {
    const pg = document.getElementById('page-data');
    if (pg) {
      const data = JSON.parse(pg.textContent || '{}');
      if (data.csrf && !CSRF) CSRF = data.csrf;
      PRELOAD = data.preload || null;
      PRELOAD_IMG_URL = data.preload_img_url || '';
    }
  } catch (e) {
    console.warn('page-data parse error', e);
  }

  // ------- refs UI -----------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const dropzone      = $('dropzone');
  const fileInput     = $('fileInput');
  const previewWrap   = $('previewWrap');
  const previewImg    = $('previewImg');
  const consent       = $('consentCheckbox');

  const mobileBar     = $('mobileBar');
  const pickGallery   = $('pickGallery');
  const openCamera    = $('openCamera');

  const predictBtn    = $('predictBtn');
  const clearBtn      = $('clearBtn');

  const resultEmpty   = $('resultEmpty');
  const resultCard    = $('resultCard');
  const resClass      = $('resClass');
  const resConf       = $('resConf');

  const btnCorrect    = $('btnCorrect');
  const btnWrong      = $('btnWrong');
  const correctionRow = $('correctionRow');
  const correctionSel = $('correctionSelect');
  const sendCorrection= $('sendCorrection');
  const resultMsg     = $('resultMsg');

  // ------- stato & costanti --------------------------------------------------
  let selectedFile = null;
  let selectedURL  = null;
  let busyOverlay  = null;
  let submitting   = false;
  const MAX_SIZE   = 10 * 1024 * 1024; // 10MB
  const MIN_W = 96, MIN_H = 96;

  // ------- device detection ---------------------------------------------------
  function isMobile() {
    const ua = navigator.userAgent || navigator.vendor || '';
    return /Android|webOS|iPhone|iPad|iPod|Windows Phone|BlackBerry/i.test(ua) || /Mobi/i.test(ua);
  }

  // Uploader congelato? (niente click/drag)
  function setUploaderFrozen(on) {
    const cls = ['pointer-events-none', 'opacity-50'];
    if (on) {
      dropzone?.classList.add(...cls);
      mobileBar?.classList.add(...cls);
    } else {
      dropzone?.classList.remove(...cls);
      mobileBar?.classList.remove(...cls);
      if (!isMobile()) dropzone?.classList.remove('hidden');
    }
  }

  // Mostra la barra Galleria/Fotocamera **solo su mobile** e nasconde la dropzone
  if (isMobile()) {
    mobileBar?.classList.remove('hidden');
    dropzone?.classList.add('hidden');
  } else {
    mobileBar?.classList.add('hidden');
  }

  // ------- helper UI ---------------------------------------------------------
  function setPredictEnabled(on) {
    if (!predictBtn) return;
    if (on) predictBtn.removeAttribute('disabled');
    else predictBtn.setAttribute('disabled', 'disabled');
  }

  function showMsg(text, colorTailwind) {
    if (!resultMsg) return;
    resultMsg.textContent = text;
    resultMsg.className = `text-sm ${colorTailwind}`;
    resultMsg.classList.remove('hidden');
  }
  function hideMsg() { resultMsg?.classList.add('hidden'); }

  function showBusy() {
    if (busyOverlay) return;
    busyOverlay = document.createElement('div');
    busyOverlay.className =
      'fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center';
    busyOverlay.innerHTML =
      '<div class="px-4 py-3 rounded-lg bg-gray-900 border border-gray-800 text-sm">Attendere…</div>';
    document.body.appendChild(busyOverlay);
  }
  function hideBusy() {
    busyOverlay?.remove();
    busyOverlay = null;
  }

  function revokeURL() {
    if (selectedURL) {
      URL.revokeObjectURL(selectedURL);
      selectedURL = null;
    }
  }

  function resetUI() {
    submitting = false;
    revokeURL();
    selectedFile = null;
    if (fileInput) fileInput.value = '';
    previewWrap?.classList.add('hidden');
    if (previewImg) previewImg.src = '';
    if (!isMobile()) dropzone?.classList.remove('hidden');
    resultEmpty?.classList.remove('hidden');
    resultCard?.classList.add('hidden');
    hideMsg();
    btnCorrect?.removeAttribute('disabled');
    btnWrong?.removeAttribute('disabled');
    btnWrong?.classList.remove('hidden');
    correctionRow?.classList.add('hidden');
    setPredictEnabled(false);
    setUploaderFrozen(false);
    window.__LAST_PREDICTION_ID__ = null;
  }

  // ---- supporto formati / validazione tipo ----------------------------------
  function isAllowedByName(name) {
    const n = (name || '').toLowerCase();
    return /\.(jpe?g|png|webp|heic|heif)$/.test(n);
  }
  function validateBasics(f) {
    if (!f) return 'Nessun file selezionato.';
    const t = (f.type || '').toLowerCase();
    const okType = (t.startsWith('image/')) || isAllowedByName(f.name);
    if (!okType) return 'Formato non supportato. Usa JPG/PNG/WebP (HEIC/HEIF ok se il server lo supporta).';
    if (f.size > MAX_SIZE) return 'File troppo grande. Max 10MB.';
    return null;
  }

  // ------- controllo dimensioni "best effort" --------------------------------
  async function checkImageDimensions(file) {
    const t = (file.type || '').toLowerCase();

    // Se HEIC/HEIF o tipo sconosciuto: lascia che sia il server a validare.
    if (!t || t === 'image/heic' || t === 'image/heif') {
      return true;
    }

    // Prova con createImageBitmap (veloce)
    try {
      if (window.createImageBitmap) {
        const bmp = await createImageBitmap(file);
        const ok = (bmp.width >= MIN_W && bmp.height >= MIN_H);
        bmp.close?.();
        return ok;
      }
    } catch (_) {
      // continua con <img>...
    }

    // Fallback con <img>. Se non riesce a decodificare, NON bloccare lato client.
    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const im = new Image();
      im.onload = () => {
        const ok = (im.naturalWidth >= MIN_W && im.naturalHeight >= MIN_H);
        URL.revokeObjectURL(url);
        resolve(ok);
      };
      im.onerror = () => { URL.revokeObjectURL(url); resolve(true); }; // <- consenti
      im.src = url;
    });
  }

  async function assignSelected(file) {
    const err = validateBasics(file);
    if (err) { showMsg(err, 'text-rose-300'); setPredictEnabled(false); return; }

    const okDim = await checkImageDimensions(file);
    if (!okDim) {
      showMsg(`Immagine troppo piccola (min ${MIN_W}×${MIN_H}px).`, 'text-rose-300');
      setPredictEnabled(false);
      return;
    }

    selectedFile = file;
    revokeURL();
    selectedURL = URL.createObjectURL(file);
    if (previewImg) previewImg.src = selectedURL;
    previewWrap?.classList.remove('hidden');
    dropzone?.classList.add('hidden'); // nascondi uploader dopo la scelta
    setPredictEnabled(true);
    hideMsg();
  }

  // ------- apertura dialog con un solo input ---------------------------------
  function openFileDialog() {
    if (!fileInput) return;
    fileInput.value = '';
    fileInput.click();
  }
  dropzone?.addEventListener('click', openFileDialog);
  previewImg?.addEventListener('click', openFileDialog);
  dropzone?.addEventListener('dblclick', openFileDialog);

  // Azioni mobile: Galleria / Fotocamera (stesso input, attributi switch)
  pickGallery?.addEventListener('click', () => {
    try {
      fileInput?.removeAttribute('capture');
      fileInput?.setAttribute('accept','image/*'); // <— include HEIC/HEIF
    } catch (_) {}
    openFileDialog();
  });
  openCamera?.addEventListener('click', () => {
    try {
      fileInput?.setAttribute('capture','environment');
      fileInput?.setAttribute('accept','image/*');
    } catch (_) {}
    openFileDialog();
  });

  // ------- drag & drop -------------------------------------------------------
  ['dragenter','dragover','dragleave','drop'].forEach(ev => {
    dropzone?.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); }, false);
  });
  ['dragenter','dragover'].forEach(ev => {
    dropzone?.addEventListener(ev, () => dropzone.classList.add('ring-2','ring-emerald-600'));
  });
  ['dragleave','drop'].forEach(ev => {
    dropzone?.addEventListener(ev, () => dropzone.classList.remove('ring-2','ring-emerald-600'));
  });
  dropzone?.addEventListener('drop', e => {
    const files = e.dataTransfer?.files;
    if (files && files.length) assignSelected(files[0]);
  });

  // ------- change input file -------------------------------------------------
  fileInput?.addEventListener('change', e => {
    const f = e.target.files && e.target.files[0];
    if (f) assignSelected(f);
  });

  // ------- incolla da clipboard (Ctrl+V) -------------------------------------
  document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items || [];
    for (const it of items) {
      if (it.type?.startsWith('image/')) {
        const f = it.getAsFile();
        if (f) { assignSelected(f); break; }
      }
    }
  });

  // ------- fetch con timeout -------------------------------------------------
  async function fetchWithTimeout(url, opts = {}, ms = 20000) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), ms);
    try {
      const resp = await fetch(url, { ...opts, signal: ctrl.signal });
      return resp;
    } finally { clearTimeout(t); }
  }
  function friendlyHTTPError(status, fallback = 'Errore') {
    if (status === 413) return 'File troppo grande (413).';
    if (status === 415) return 'Formato non supportato (415).';
    if (status === 429) return 'Troppe richieste (429). Riprova tra poco.';
    return `${fallback} (HTTP ${status}).`;
  }

  // ------- Predici -----------------------------------------------------------
  predictBtn?.addEventListener('click', async () => {
    if (submitting) return;
    if (!selectedFile) { showMsg("Seleziona un'immagine.", 'text-rose-300'); return; }

    submitting = true;
    setPredictEnabled(false);
    hideMsg(); showBusy();

    try {
      const fd = new FormData();
      fd.append('file', selectedFile, selectedFile.name || 'image');
      fd.append('consent_training', consent?.checked ? 'on' : 'off');

      const r = await fetchWithTimeout('/predict', {
        method: 'POST',
        headers: CSRF ? { 'X-CSRFToken': CSRF } : undefined,
        body: fd
      }, 25000);

      let js = {};
      try { js = await r.json(); } catch (_) { js = {}; }

      if (!r.ok) {
        const msg = js.error || friendlyHTTPError(r.status, 'Errore durante la predizione');
        throw new Error(msg);
      }

      resultEmpty?.classList.add('hidden');
      resultCard?.classList.remove('hidden');
      if (resClass) resClass.textContent = js.class;
      if (resConf)  resConf.textContent  = js.confidence;

      btnCorrect?.removeAttribute('disabled');
      btnWrong?.removeAttribute('disabled');
      btnWrong?.classList.remove('hidden');
      correctionRow?.classList.add('hidden');
      hideMsg();

      window.__LAST_PREDICTION_ID__ = js.prediction_id;
    } catch (err) {
      console.error('predict error', err);
      showMsg(err.message || 'Errore durante la predizione', 'text-rose-300');
    } finally {
      hideBusy();
      submitting = false;
      setPredictEnabled(!!selectedFile);
    }
  });

  // ------- Pulisci -----------------------------------------------------------
  clearBtn?.addEventListener('click', resetUI);

  // ------- Feedback: corretta ------------------------------------------------
  btnCorrect?.addEventListener('click', async () => {
    const pid = window.__LAST_PREDICTION_ID__;
    if (!pid || submitting) return;

    submitting = true; showBusy();
    btnCorrect.setAttribute('disabled', 'disabled');
    try {
      const r = await fetchWithTimeout('/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(CSRF ? { 'X-CSRFToken': CSRF } : {})
        },
        body: JSON.stringify({ prediction_id: pid, feedback_type: 'correct' })
      }, 20000);

      let js = {};
      try { js = await r.json(); } catch (_) { js = {}; }

      if (!r.ok) {
        const msg = js.error || friendlyHTTPError(r.status, 'Errore');
        throw new Error(msg);
      }

      showMsg('Grazie! Predizione confermata.', 'text-emerald-400');
      btnWrong?.classList.add('hidden');
      btnWrong?.setAttribute('disabled','disabled');

      window.__LAST_PREDICTION_ID__ = null;
      setUploaderFrozen(false);
      setPredictEnabled(!!selectedFile);
    } catch (err) {
      console.error('feedback/correct error', err);
      showMsg('Errore: ' + (err.message || err), 'text-rose-300');
    } finally {
      hideBusy(); submitting = false;
      btnCorrect.removeAttribute('disabled');
    }
  });

  // ------- Feedback: sbagliata -> correzione --------------------------------
  btnWrong?.addEventListener('click', () => {
    btnWrong.classList.add('hidden');
    correctionRow?.classList.remove('hidden');
    hideMsg();
  });

  // ------- Invia correzione --------------------------------------------------
  sendCorrection?.addEventListener('click', async () => {
    const pid = window.__LAST_PREDICTION_ID__;
    if (!pid || submitting) return;

    submitting = true; showBusy();
    sendCorrection.setAttribute('disabled','disabled');
    try {
      const r = await fetchWithTimeout('/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(CSRF ? { 'X-CSRFToken': CSRF } : {})
        },
        body: JSON.stringify({
          prediction_id: pid,
          feedback_type: 'incorrect',
          correct_class: correctionSel?.value
        })
      }, 20000);

      let js = {};
      try { js = await r.json(); } catch (_) { js = {}; }

      if (!r.ok) {
        const msg = js.error || friendlyHTTPError(r.status, 'Errore');
        throw new Error(msg);
      }

      showMsg('Correzione inviata. Grazie!', 'text-rose-300');
      btnCorrect?.setAttribute('disabled','disabled');
      sendCorrection.setAttribute('disabled','disabled');

      window.__LAST_PREDICTION_ID__ = null;
      setUploaderFrozen(false);
      setPredictEnabled(!!selectedFile);
    } catch (err) {
      console.error('feedback/incorrect error', err);
      showMsg('Errore: ' + (err.message || err), 'text-rose-300');
      sendCorrection.removeAttribute('disabled');
    } finally {
      hideBusy(); submitting = false;
    }
  });

  // ------- Preload: predizione pendente -------------------------------------
  if (PRELOAD) {
    resultEmpty?.classList.add('hidden');
    resultCard?.classList.remove('hidden');
    window.__LAST_PREDICTION_ID__ = PRELOAD.id;
    if (resClass) resClass.textContent = PRELOAD.cls || '—';
    if (resConf)  resConf.textContent  = PRELOAD.conf || '—';

    if (PRELOAD_IMG_URL) {
      previewImg.src = PRELOAD_IMG_URL;
      previewWrap?.classList.remove('hidden');
    }

    setPredictEnabled(false);
    setUploaderFrozen(true);
  } else {
    resetUI();
  }

  // ------- scorciatoie tastiera ----------------------------------------------
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      if (!predictBtn?.disabled) { e.preventDefault(); predictBtn.click(); }
    } else if (e.key === 'Escape') {
      e.preventDefault(); clearBtn?.click();
    }
  });

  // Default: su mobile preferiamo Galleria se l'utente tocca la dropzone,
  // ma lasciamo il bottone "Scatta foto" per camera diretta.
  if (isMobile()) {
    try { fileInput?.removeAttribute('capture'); fileInput?.setAttribute('accept','image/*'); } catch (_) {}
  } else {
    try { fileInput?.removeAttribute('capture'); fileInput?.setAttribute('accept','image/jpeg,image/png,image/webp'); } catch (_) {}
  }

  // Rilascia l’URL blob quando lasci la pagina
  window.addEventListener('beforeunload', () => {
    try { if (selectedURL) URL.revokeObjectURL(selectedURL); } catch (_) {}
  });
})();
