(function () {
  console.log('[index.js] loaded');

  // ---------------------------------------------------------------------------
  // 1) Recupero CSRF e "preload" (predizione pendente) dal template
  // ---------------------------------------------------------------------------
  let CSRF = '';
  try {
    // Se il template espone un <meta name="csrf-token" content="..."> lo uso
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta?.content) CSRF = meta.content;
  } catch (_) {}

  let PRELOAD = null;         // { id, cls, conf } oppure null
  let PRELOAD_IMG_URL = '';   // URL dell'immagine della predizione pendente (opzionale)
  try {
    // Il template dovrebbe avere un <script type="application/json" id="page-data">...</script>
    const pg = document.getElementById('page-data');
    if (pg) {
      const data = JSON.parse(pg.textContent || '{}');
      // Se il JSON contiene 'csrf' ed io non l'ho già preso dal meta, lo uso
      if (data.csrf && !CSRF) CSRF = data.csrf;
      PRELOAD = data.preload || null;
      PRELOAD_IMG_URL = data.preload_img_url || '';
    }
  } catch (e) {
    console.warn('page-data parse error', e);
  }

  // ---------------------------------------------------------------------------
  // 2) Shortcuts ai nodi DOM (by id) e riferimenti alla UI
  // ---------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);

  // Area upload / anteprima
  const dropzone      = $('dropzone');
  const fileInput     = $('fileInput');
  const previewWrap   = $('previewWrap');
  const previewImg    = $('previewImg');
  const consent       = $('consentCheckbox');

  // Barra "mobile" (solo smartphone)
  const mobileBar     = $('mobileBar');
  const pickGallery   = $('pickGallery');
  const openCamera    = $('openCamera');

  // Pulsanti principali
  const predictBtn    = $('predictBtn');
  const clearBtn      = $('clearBtn');

  // Card risultato
  const resultEmpty   = $('resultEmpty');
  const resultCard    = $('resultCard');
  const resClass      = $('resClass');
  const resConf       = $('resConf');

  // Feedback
  const btnCorrect    = $('btnCorrect');
  const btnWrong      = $('btnWrong');
  const correctionRow = $('correctionRow');
  const correctionSel = $('correctionSelect');
  const sendCorrection= $('sendCorrection');
  const resultMsg     = $('resultMsg');

  // ---------------------------------------------------------------------------
  // 3) Stato runtime e costanti
  // ---------------------------------------------------------------------------
  let selectedFile = null;     // File scelto dall'utente
  let selectedURL  = null;     // URL blob per l'anteprima (da revocare)
  let busyOverlay  = null;     // overlay "Attendere…"
  let submitting   = false;    // previene click/richieste multiple
  const MAX_SIZE   = 10 * 1024 * 1024; // 10MB
  const MIN_W = 96, MIN_H = 96;        // dimensioni minime ragionevoli

  // ---------------------------------------------------------------------------
  // 4) Rilevamento "mobile" e setup UI coerente
  // ---------------------------------------------------------------------------
  function isMobile() {
    // Heuristica semplice: sufficiente allo scopo (Android/iOS)
    const ua = navigator.userAgent || navigator.vendor || '';
    return /Android|webOS|iPhone|iPad|iPod|Windows Phone|BlackBerry/i.test(ua) || /Mobi/i.test(ua);
  }

  // Quando la UI è "congelata" non voglio che l'utente possa cliccare/trascinare
  function setUploaderFrozen(on) {
    const cls = ['pointer-events-none', 'opacity-50'];
    if (on) {
      dropzone?.classList.add(...cls);
      mobileBar?.classList.add(...cls);
    } else {
      dropzone?.classList.remove(...cls);
      mobileBar?.classList.remove(...cls);
      if (!isMobile()) dropzone?.classList.remove('hidden'); // su desktop torna visibile
    }
  }

  // Scelta UX: su **mobile** mostro SOLO i due bottoni (Galleria / Scatta foto);
  // su **desktop** lascio la dropzone per drag&drop/click.
  if (isMobile()) {
    mobileBar?.classList.remove('hidden');
    dropzone?.classList.add('hidden');
  } else {
    mobileBar?.classList.add('hidden');
  }

  // ---------------------------------------------------------------------------
  // 5) Piccole utility UI (abilitazione pulsanti, messaggi, overlay)
  // ---------------------------------------------------------------------------
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

  // Revoca l’URL blob quando non serve più, per non accumulare in memoria
  function revokeURL() {
    if (selectedURL) {
      URL.revokeObjectURL(selectedURL);
      selectedURL = null;
    }
  }

  // Ripristina la UI come all'inizio
  function resetUI() {
    submitting = false;
    revokeURL();
    selectedFile = null;
    if (fileInput) fileInput.value = '';
    previewWrap?.classList.add('hidden');
    if (previewImg) previewImg.src = '';
    if (!isMobile()) dropzone?.classList.remove('hidden'); // su desktop torna visibile
    resultEmpty?.classList.remove('hidden');
    resultCard?.classList.add('hidden');
    hideMsg();
    btnCorrect?.removeAttribute('disabled');
    btnWrong?.removeAttribute('disabled');
    btnWrong?.classList.remove('hidden');
    correctionRow?.classList.add('hidden');
    setPredictEnabled(false);
    setUploaderFrozen(false);
    window.__LAST_PREDICTION_ID__ = null; // id della predizione corrente (se presente)
  }

  // ---------------------------------------------------------------------------
  // 6) Validazione lato client (best-effort, il server resta sorgente di verità)
  // ---------------------------------------------------------------------------
  // Controllo tipo/estensione: accetto standard + HEIC/HEIF per lasciare al server la decodifica
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

  // Verifica dimensioni minime (se possibile). Per HEIC/HEIF lascio passare e farà fede il controllo server.
  async function checkImageDimensions(file) {
    const t = (file.type || '').toLowerCase();

    // HEIC/HEIF o tipo non noto: non blocco lato client
    if (!t || t === 'image/heic' || t === 'image/heif') {
      return true;
    }

    // API moderna e veloce
    try {
      if (window.createImageBitmap) {
        const bmp = await createImageBitmap(file);
        const ok = (bmp.width >= MIN_W && bmp.height >= MIN_H);
        bmp.close?.();
        return ok;
      }
    } catch (_) {
      // se fallisce, passo al fallback <img>
    }

    // Fallback con tag <img>. Se non riesco a leggere, non blocco.
    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const im = new Image();
      im.onload = () => {
        const ok = (im.naturalWidth >= MIN_W && im.naturalHeight >= MIN_H);
        URL.revokeObjectURL(url);
        resolve(ok);
      };
      im.onerror = () => { URL.revokeObjectURL(url); resolve(true); };
      im.src = url;
    });
  }

  // Quando l’utente sceglie un file, preparo anteprima e abilito "Predici"
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
    selectedURL = URL.createObjectURL(file);      // URL blob per <img>
    if (previewImg) previewImg.src = selectedURL; // anteprima
    previewWrap?.classList.remove('hidden');
    dropzone?.classList.add('hidden');            // nascondo uploader finché c'è un file
    setPredictEnabled(true);
    hideMsg();
  }

  // ---------------------------------------------------------------------------
  // 7) Modi di selezione file: click, drag&drop, clipboard, mobile actions
  // ---------------------------------------------------------------------------
  function openFileDialog() {
    if (!fileInput) return;
    fileInput.value = ''; // consente riselezione dello stesso file
    fileInput.click();
  }
  dropzone?.addEventListener('click', openFileDialog);
  previewImg?.addEventListener('click', openFileDialog);
  dropzone?.addEventListener('dblclick', openFileDialog);

  // Bottoni "Galleria" / "Scatta foto" su mobile (usano lo stesso <input type=file>)
  pickGallery?.addEventListener('click', () => {
    try {
      fileInput?.removeAttribute('capture');      // niente camera forzata
      fileInput?.setAttribute('accept','image/*'); // accetta tutto (anche HEIC)
    } catch (_) {}
    openFileDialog();
  });
  openCamera?.addEventListener('click', () => {
    try {
      fileInput?.setAttribute('capture','environment'); // preferisci camera posteriore
      fileInput?.setAttribute('accept','image/*');
    } catch (_) {}
    openFileDialog();
  });

  // Drag & drop (desktop)
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

  // Cambio file via <input>
  fileInput?.addEventListener('change', e => {
    const f = e.target.files && e.target.files[0];
    if (f) assignSelected(f);
  });

  // Incolla da clipboard (desktop): Ctrl+V di un’immagine
  document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items || [];
    for (const it of items) {
      if (it.type?.startsWith('image/')) {
        const f = it.getAsFile();
        if (f) { assignSelected(f); break; }
      }
    }
  });

  // ---------------------------------------------------------------------------
  // 8) fetch con timeout + mappatura errori HTTP "amica"
  // ---------------------------------------------------------------------------
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

  // ---------------------------------------------------------------------------
  // 9) Predizione: invio al backend /predict
  // ---------------------------------------------------------------------------
  predictBtn?.addEventListener('click', async () => {
    if (submitting) return;
    if (!selectedFile) { showMsg("Seleziona un'immagine.", 'text-rose-300'); return; }

    submitting = true;
    setPredictEnabled(false);
    hideMsg(); showBusy();

    try {
      const fd = new FormData();
      fd.append('file', selectedFile, selectedFile.name || 'image');            // immagine
      fd.append('consent_training', consent?.checked ? 'on' : 'off');           // consenso

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

      // Mostro il risultato
      resultEmpty?.classList.add('hidden');
      resultCard?.classList.remove('hidden');
      if (resClass) resClass.textContent = js.class;
      if (resConf)  resConf.textContent  = js.confidence;

      // Abilito i bottoni feedback
      btnCorrect?.removeAttribute('disabled');
      btnWrong?.removeAttribute('disabled');
      btnWrong?.classList.remove('hidden');
      correctionRow?.classList.add('hidden');
      hideMsg();

      // Memorizzo l'id della predizione restituito dal backend
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

  // ---------------------------------------------------------------------------
  // 10) Pulisci: torna allo stato iniziale
  // ---------------------------------------------------------------------------
  clearBtn?.addEventListener('click', resetUI);

  // ---------------------------------------------------------------------------
  // 11) Feedback: "È corretta"
  // ---------------------------------------------------------------------------
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

      // Predizione chiusa → scongela uploader
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

  // ---------------------------------------------------------------------------
  // 12) Feedback: "È sbagliata" → mostra select di correzione
  // ---------------------------------------------------------------------------
  btnWrong?.addEventListener('click', () => {
    btnWrong.classList.add('hidden');
    correctionRow?.classList.remove('hidden');
    hideMsg();
  });

  // ---------------------------------------------------------------------------
  // 13) Invia correzione con la classe scelta
  // ---------------------------------------------------------------------------
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

      // Predizione chiusa → scongela uploader
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

  // ---------------------------------------------------------------------------
  // 14) Stato "preload": se il server ci dice che c'è una predizione pendente,
  //     visualizzala e "congela" l'uploader finché non arriva un feedback.
  // ---------------------------------------------------------------------------
  if (PRELOAD) {
    resultEmpty?.classList.add('hidden');
    resultCard?.classList.remove('hidden');
    window.__LAST_PREDICTION_ID__ = PRELOAD.id;
    if (resClass) resClass.textContent = PRELOAD.cls || '—';
    if (resConf)  resConf.textContent  = PRELOAD.conf || '—';

    if (PRELOAD_IMG_URL) {
      // Mostro l'immagine associata alla predizione pendente
      previewImg.src = PRELOAD_IMG_URL;
      previewWrap?.classList.remove('hidden');
    }

    setPredictEnabled(false);  // non permettere una nuova predizione
    setUploaderFrozen(true);   // blocca upload finché non si chiude la precedente
  } else {
    resetUI();
  }

  // ---------------------------------------------------------------------------
  // 15) Scorciatoie tastiera (desktop)
  // ---------------------------------------------------------------------------
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      if (!predictBtn?.disabled) { e.preventDefault(); predictBtn.click(); }
    } else if (e.key === 'Escape') {
      e.preventDefault(); clearBtn?.click();
    }
  });

  // ---------------------------------------------------------------------------
  // 16) Impostazioni finali sugli attributi dell'input file
  // ---------------------------------------------------------------------------
  if (isMobile()) {
    // Su mobile, preferisco non forzare formati; lascia scegliere alla UI Android/iOS
    try { fileInput?.removeAttribute('capture'); fileInput?.setAttribute('accept','image/*'); } catch (_) {}
  } else {
    // Su desktop limito a tipi più comuni
    try { fileInput?.removeAttribute('capture'); fileInput?.setAttribute('accept','image/jpeg,image/png,image/webp'); } catch (_) {}
  }

  // ---------------------------------------------------------------------------
  // 17) Pulizia: libera l'URL blob quando lasci la pagina
  // ---------------------------------------------------------------------------
  window.addEventListener('beforeunload', () => {
    try { if (selectedURL) URL.revokeObjectURL(selectedURL); } catch (_) {}
  });
})();
