const $ = id => document.getElementById(id);
const keys = ['stick', 'cpu', 'mps'];
const state = {ready: false, busy: false, remoteBusy: false, live: false, stream: null,
  cameraPending: false, rendering: false, liveRunning: false,
  input: {example: 'normal'}, result: null, frame: '/example/normal'};
const fmt = (x, n = 1) => Number(x).toLocaleString('fr-FR', {minimumFractionDigits: n, maximumFractionDigits: n});

function controls() {
  const busy = state.busy || state.remoteBusy || state.rendering;
  for (const id of ['compare', 'reference', 'reset-reference']) $(id).disabled = !state.ready || busy || state.liveRunning;
  $('live').disabled = !state.ready || !state.stream || state.cameraPending || ((busy || state.liveRunning) && !state.live);
  $('live').textContent = state.live ? 'Arrêter le live' : 'Live sur le stick';
  $('camera-toggle').disabled = busy || state.liveRunning || state.cameraPending;
  $('camera-toggle').textContent = state.cameraPending ? 'Autorisation caméra…' : state.stream ? 'Éteindre la caméra' : 'Activer la caméra';
  $('file').disabled = busy || state.liveRunning;
  document.querySelectorAll('[data-example]').forEach(x => { x.disabled = busy || state.liveRunning; });
  $('export').disabled = !state.result;
}

async function request(path, options) {
  const response = await fetch(path, options), data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Requête impossible');
  return data;
}

async function status() {
  try {
    const data = await request('/api/status');
    state.ready = data.ready; state.remoteBusy = data.busy;
    if (!state.busy) $('status').textContent = data.error || data.progress;
    $('status-dot').style.background = data.error ? '#a33d27' : data.ready ? '#236647' : '#aa8d55';
    for (const key of keys) {
      const item = data.backends?.[key];
      $('state-' + key).textContent = item?.ready ? 'Prêt' : item ? 'Indisponible' : 'Chargement';
      $('state-' + key).title = item?.error || '';
    }
    if (data.reference && $('reference-label').textContent !== data.reference) {
      $('reference-label').textContent = data.reference;
      $('reference-image').src = '/api/reference?' + Date.now();
    }
  } catch (error) {
    state.ready = false;
    $('status').textContent = 'Serveur local inaccessible. Relancer python -m scripts.serve_dino.';
  }
  controls();
}

function error(message) { $('error').textContent = message; $('error').hidden = !message; }

function clearResults() {
  state.result = null;
  for (const key of keys) {
    $('score-' + key).textContent = '—'; $('latency-' + key).textContent = '—';
    const canvas = $('map-' + key), ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  controls();
}

function stopCamera() {
  state.live = false;
  state.stream?.getTracks().forEach(track => track.stop()); state.stream = null;
  $('camera').srcObject = null; $('camera').hidden = true; $('preview').hidden = false;
  $('source-label').textContent = state.input.example ? 'MVTec AD / bottle' : 'Image locale';
  controls();
}

function square(source) {
  const width = source.videoWidth || source.naturalWidth, height = source.videoHeight || source.naturalHeight;
  if (!width || !height) throw new Error('La caméra n’a pas encore fourni d’image.');
  const side = Math.min(width, height), canvas = document.createElement('canvas');
  canvas.width = canvas.height = 224;
  canvas.getContext('2d').drawImage(source, (width - side) / 2, (height - side) / 2, side, side, 0, 0, 224, 224);
  return canvas.toDataURL('image/png');
}

function capture() {
  if (!state.stream) return {payload: {...state.input}, frame: state.frame};
  const frame = square($('camera'));
  return {payload: {image: frame}, frame};
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image); image.onerror = () => reject(new Error('Image illisible.')); image.src = src;
  });
}

async function drawResults(data, frame) {
  const image = await loadImage(frame), base = square(image), cropped = await loadImage(base);
  state.result = data;
  for (const key of keys) {
    const result = data.results[key], canvas = $('map-' + key), ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 224, 224);
    if (!result || result.error) {
      $('score-' + key).textContent = '—'; $('latency-' + key).textContent = '—';
      if (result?.error) $('state-' + key).textContent = result.error;
      continue;
    }
    ctx.drawImage(cropped, 0, 0, 224, 224);
    const heat = document.createElement('canvas'); heat.width = heat.height = 16;
    const hc = heat.getContext('2d'), pixels = hc.createImageData(16, 16);
    result.patch_distances.flat().forEach((value, i) => {
      const t = Math.max(0, Math.min(1, value / 1.2)), offset = i * 4;
      pixels.data[offset] = t < .5 ? 44 + 2 * t * 198 : 242 - (t - .5) * 28;
      pixels.data[offset + 1] = t < .5 ? 97 + 2 * t * 108 : 205 - (t - .5) * 250;
      pixels.data[offset + 2] = t < .5 ? 132 - 2 * t * 56 : 76 - (t - .5) * 62;
      pixels.data[offset + 3] = 115;
    });
    hc.putImageData(pixels, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(heat, 0, 0, 224, 224);
    $('score-' + key).textContent = fmt(result.score, 3);
    $('latency-' + key).textContent = fmt(result.median_ms);
  }
  $('measurement-note').textContent = `${data.runs === 3 ? 'Médiane de trois passages' : 'Un passage par image'} ; chargement, calibration et préparation exclus. Score = distance, pas probabilité. Cartes 16 × 16 agrandies pour l’affichage, avec une échelle identique et fixe pour les trois moteurs.`;
}

async function job(kind, payload) {
  state.busy = true; controls(); error('');
  try {
    const started = await request('/api/run', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({kind, ...payload})});
    for (;;) {
      const current = await request('/api/jobs/' + started.id); $('status').textContent = current.progress;
      if (current.state === 'error') throw new Error(current.error);
      if (current.state === 'complete') return current.result;
      await new Promise(resolve => setTimeout(resolve, 150));
    }
  } finally { state.busy = false; state.remoteBusy = false; controls(); }
}

async function compare(live = false) {
  state.rendering = true; controls();
  try {
    const {payload, frame} = capture();
    const result = await job('compare', {...payload, backend: live ? 'stick' : 'all', runs: live ? 1 : 3});
    await drawResults(result, frame);
    $('status').textContent = live ? 'Live sur le stick · une image à la fois' : 'Comparaison terminée.';
  } finally { state.rendering = false; controls(); }
}

async function liveLoop() {
  state.liveRunning = true; controls();
  try {
    while (state.live) await compare(true);
  } catch (e) { error(e.message); }
  finally { state.live = false; state.liveRunning = false; controls(); }
}

$('compare').addEventListener('click', () => compare().catch(e => error(e.message)));
$('live').addEventListener('click', () => {
  state.live = !state.live; controls(); if (state.live) liveLoop();
});
$('camera-toggle').addEventListener('click', async () => {
  if (state.stream) { stopCamera(); return; }
  state.cameraPending = true; controls();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Caméra indisponible dans ce navigateur. Ouvrir cette adresse dans Chrome ou Safari, ou choisir une image.');
    state.stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}}, audio: false});
    $('camera').srcObject = state.stream; await $('camera').play();
    $('camera').hidden = false; $('preview').hidden = true; $('source-label').textContent = 'Caméra · recadrage carré central';
    error(''); clearResults();
  } catch (e) { stopCamera(); error('Accès caméra : ' + e.message); }
  state.cameraPending = false; controls();
});
async function calibrate(payload) {
  const result = await job('reference', payload); clearResults(); await status();
  $('reference-image').src = '/api/reference?' + Date.now();
  $('status').textContent = 'Nouvelle référence prête.';
  const failures = Object.entries(result.errors); if (failures.length) error(failures.map(([k, v]) => `${k} : ${v}`).join(' · '));
}
$('reference').addEventListener('click', () => {
  try { calibrate(capture().payload).catch(e => error(e.message)); } catch (e) { error(e.message); }
});
$('reset-reference').addEventListener('click', () => calibrate({example: 'reference'}).catch(e => error(e.message)));
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  stopCamera(); state.input = {example: button.dataset.example}; state.frame = '/example/' + button.dataset.example;
  $('preview').src = state.frame; $('source-label').textContent = 'MVTec AD / bottle'; clearResults();
}));
$('file').addEventListener('change', async () => {
  const file = $('file').files[0]; if (!file) return;
  try {
    stopCamera(); const url = URL.createObjectURL(file);
    try { const image = await loadImage(url); state.frame = square(image); }
    finally { URL.revokeObjectURL(url); }
    state.input = {image: state.frame}; $('preview').src = state.frame; $('source-label').textContent = 'Image locale'; clearResults(); error('');
  } catch (e) { error(e.message); }
});
$('export').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(state.result, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'dino-local-results.json';
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 30000);
});
window.addEventListener('pagehide', stopCamera);
status(); setInterval(() => { if (!state.busy) status(); }, 2000);
