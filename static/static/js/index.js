(function () {
  console.log('[index.js] loaded');

  // --- leggi dati (CSRF + preload) ---
  let CSRF = '';
  try {
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta && meta.content) CSRF = meta.content;
  } catch (_) {}
  let PRELOAD = null;
  try {
    const pg = document.getElementById('page-data');
    if (pg) {
      const data = JSON.parse(pg.textContent || '{}');
      if (data.csrf && !CSRF) CSRF = data.csrf;
      PRELOAD = data.preload || null;
    }
  } catch (e) { console.warn('page-data parse error', e); }

  // --- refs UI ---
  const $ = (id) => document.getElementById(id);
  const dropzone      = $('dropzone');
  const fileInput     = $('fileInput');
  const previewWrap   = $('previewWrap');
  const previewImg    = $('previewImg');
  const consent       = $('consentCheckbox');

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

  let currentPredictionId = null;

  // --- helpers ---
  function resetUI() {
    fileInput.value = "";
    previewWrap.classList.add('hidden');
    previewImg.src = "";
    dropzone.classList.remove('hidden');          // <— ri-mostro la dropzone
    resultEmpty.classList.remove('hidden');
    resultCard.classList.add('hidden');
    resultMsg.classList.add('hidden');
    resultMsg.textContent = "";
    btnCorrect?.removeAttribute('disabled');
    btnWrong?.removeAttribute('disabled');
    btnWrong?.classList.remove('hidden');
    correctionRow?.classList.add('hidden');
    currentPredictionId = null;
  }

  function showPreview(file) {
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewWrap.classList.remove('hidden');
    dropzone.classList.add('hidden');            // <— nascondo la dropzone
  }

  function setBusy(el, busy=true) {
    if (!el) return;
    if (busy) el.setAttribute('disabled','disabled');
    else el.removeAttribute('disabled');
  }

  // --- dropzone ---
  dropzone?.addEventListener('click', () => fileInput?.click());
  previewImg?.addEventListener('click', () => fileInput?.click());
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
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      fileInput.files = files;
      showPreview(files[0]);
    }
  });
  fileInput?.addEventListener('change', e => {
    const f = e.target.files && e.target.files[0];
    if (f) showPreview(f);
  });

  // --- predict ---
  predictBtn?.addEventListener('click', async () => {
    const f = fileInput?.files && fileInput.files[0];
    if (!f) { alert("Seleziona un'immagine."); return; }

    setBusy(predictBtn, true);
    resultMsg.classList.add('hidden');

    try {
      const fd = new FormData();
      fd.append('file', f);
      fd.append('consent_training', consent?.checked ? 'on' : 'off');

      const r = await fetch('/predict', {
        method: 'POST',
        headers: CSRF ? { 'X-CSRFToken': CSRF } : undefined,
        body: fd
      });
      const js = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(js.error || ('HTTP '+r.status));

      currentPredictionId = js.prediction_id;
      resClass.textContent = js.class;
      resConf.textContent  = js.confidence;

      resultEmpty.classList.add('hidden');
      resultCard.classList.remove('hidden');

      btnCorrect?.removeAttribute('disabled');
      btnWrong?.removeAttribute('disabled');
      btnWrong?.classList.remove('hidden');
      correctionRow?.classList.add('hidden');
      resultMsg.classList.add('hidden');
    } catch (err) {
      console.error('predict error', err);
      alert('Errore durante la predizione: ' + err.message);
    } finally {
      setBusy(predictBtn, false);
    }
  });

  // --- clear ---
  clearBtn?.addEventListener('click', resetUI);

  // --- feedback: correct ---
  btnCorrect?.addEventListener('click', async () => {
    if (!currentPredictionId) return;
    setBusy(btnCorrect, true);
    try {
      const r = await fetch('/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(CSRF ? { 'X-CSRFToken': CSRF } : {})
        },
        body: JSON.stringify({ prediction_id: currentPredictionId, feedback_type: 'correct' })
      });
      const js = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(js.error || ('HTTP '+r.status));

      resultMsg.textContent = 'Grazie! Predizione confermata.';
      resultMsg.className = 'text-sm text-emerald-400';
      resultMsg.classList.remove('hidden');

      btnWrong?.classList.add('hidden'); // rimuovo il bottone rosso
      setBusy(btnWrong, true);
    } catch (err) {
      console.error('feedback/correct error', err);
      alert('Errore: ' + err.message);
    } finally {
      setBusy(btnCorrect, false);
    }
  });

  // --- feedback: wrong -> mostra solo la correzione ---
  btnWrong?.addEventListener('click', () => {
    btnWrong?.classList.add('hidden');
    correctionRow?.classList.remove('hidden');
    resultMsg.classList.add('hidden');
  });

  // --- invia correzione ---
  sendCorrection?.addEventListener('click', async () => {
    if (!currentPredictionId) return;
    const corr = correctionSel?.value;
    setBusy(sendCorrection, true);
    try {
      const r = await fetch('/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(CSRF ? { 'X-CSRFToken': CSRF } : {})
        },
        body: JSON.stringify({
          prediction_id: currentPredictionId,
          feedback_type: 'incorrect',
          correct_class: corr
        })
      });
      const js = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(js.error || ('HTTP '+r.status));

      resultMsg.textContent = 'Correzione inviata. Grazie!';
      resultMsg.className = 'text-sm text-rose-300';
      resultMsg.classList.remove('hidden');

      setBusy(btnCorrect, true);
      setBusy(sendCorrection, true);
    } catch (err) {
      console.error('feedback/incorrect error', err);
      alert('Errore: ' + err.message);
      setBusy(sendCorrection, false);
    }
  });

  // --- preload (predizione pendente) ---
  if (PRELOAD) {
    resultEmpty.classList.add('hidden');
    resultCard.classList.remove('hidden');
    currentPredictionId = PRELOAD.id;
    resClass.textContent = PRELOAD.cls || '—';
    resConf.textContent  = PRELOAD.conf || '—';
  } else {
    resetUI();
  }
})();
