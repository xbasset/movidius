const $ = id => document.getElementById(id);
const keys = ['stick', 'cpu', 'mps'];
const names = {stick: 'Stick · Mac accompagnant', cpu: 'CPU', mps: 'GPU Metal'};
const state = {ready: false, busy: false, remoteBusy: false, energyAvailable: false, comparison: null, energy: null};
const fmt = (value, digits = 1) => Number(value).toLocaleString('fr-FR', {maximumFractionDigits: digits, minimumFractionDigits: digits});
const percent = value => value === 0 ? '0 %' : value < .00001 ? '< 0,001 %' : fmt(value * 100, value < .01 ? 3 : 2) + ' %';
const human = label => label.replaceAll('_', ' ');

function buttons() {
  const busy = state.busy || state.remoteBusy;
  $('compare').disabled = !state.ready || busy;
  $('energy').disabled = !state.ready || busy || !state.energyAvailable;
  $('reconnect').disabled = busy;
  $('export').disabled = !state.comparison && !state.energy;
  $('message').disabled = busy;
  document.querySelectorAll('[data-example]').forEach(button => { button.disabled = busy; });
}

async function request(path, options) {
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || 'Requête impossible');
  return value;
}

async function refreshStatus() {
  try {
    const status = await request('/api/status');
    state.ready = status.ready;
    state.remoteBusy = status.busy;
    state.energyAvailable = Boolean(status.energy?.available);
    if (!state.busy) $('status').textContent = status.error || status.progress;
    $('status-dot').style.background = status.error ? '#a33d27' : status.ready ? '#146947' : '#b79552';
    for (const key of keys) {
      const backend = status.backends?.[key];
      $('state-' + key).textContent = backend ? backend.ready ? 'Prêt' : 'Indisponible' : 'Chargement';
      $('state-' + key).title = backend?.error || '';
    }
    $('reconnect').hidden = !status.ready || status.backends?.stick?.ready;
    $('energy-source').textContent = status.energy?.available ? 'IOReport · disponible' : status.ready ? 'Compteurs indisponibles' : 'IOReport · vérification…';
    if (status.energy?.error) $('energy-note').textContent = status.energy.error;
    buttons();
  } catch (error) {
    $('status').textContent = 'Serveur local inaccessible. Relancer python -m scripts.serve_crivo.';
    state.ready = false; buttons();
  }
}

function showComparison(data) {
  state.comparison = data;
  $('token-count').textContent = `${data.tokens} / 960 tokens · 77 intentions`;
  for (const key of keys) {
    const result = data.results[key];
    if (result.error) {
      $('decision-' + key).textContent = result.error;
      $('latency-' + key).textContent = '—';
      $('bar-' + key).style.width = '0%';
      $('confidence-' + key).textContent = 'Aucune inférence';
      continue;
    }
    $('latency-' + key).textContent = fmt(result.median_ms);
    $('decision-' + key).textContent = human(result.decision);
    $('bar-' + key).style.width = `${Math.min(100, result.probability * 100)}%`;
    $('confidence-' + key).textContent = percent(result.probability) + ' · ' + result.precision;
  }
  const available = keys.filter(key => !data.results[key].error);
  const agree = new Set(available.map(key => data.results[key].decision)).size === 1;
  $('comparison-note').textContent = `${agree ? 'Même décision pour les backends disponibles.' : 'Les backends ne choisissent pas tous la même intention.'} Médiane de 3 passages ; tokenisation ${fmt(data.tokenization_ms)} ms exclue. Stick : 960 tokens masqués ; Mac : ${data.tokens} tokens. Chargement exclu.`;
  renderProbabilities();
}

function renderProbabilities() {
  if (!state.comparison) return;
  const data = state.comparison;
  let indices = data.labels.map((_, i) => i);
  indices.sort((a, b) => Math.max(...keys.map(k => data.results[k].probabilities?.[b] || 0)) - Math.max(...keys.map(k => data.results[k].probabilities?.[a] || 0)));
  if (!$('all-labels').checked) indices = indices.slice(0, 10);
  $('probabilities').replaceChildren(...indices.map(i => {
    const row = document.createElement('tr');
    const label = document.createElement('td'); label.textContent = human(data.labels[i]); row.append(label);
    for (const key of keys) {
      const cell = document.createElement('td');
      cell.className = key + '-text';
      cell.textContent = data.results[key].probabilities ? percent(data.results[key].probabilities[i]) : '—';
      row.append(cell);
    }
    return row;
  }));
}

function showEnergy(data) {
  state.energy = data;
  const rows = [];
  for (const key of ['cpu', 'mps', 'stick']) {
    const sample = data.results[key];
    if (sample.error) { $('energy-detail-' + key).textContent = sample.error; continue; }
    const value = document.createTextNode(fmt(sample.gross_j_per_inference * 1000) + ' ');
    const unit = document.createElement('small'); unit.textContent = 'mJ / décision';
    $('energy-' + key).replaceChildren(value, unit);
    const net = sample.incremental_j_per_inference * 1000;
    $('energy-detail-' + key).textContent = `${fmt(sample.sum_w, 2)} W · ${sample.inferences} décisions en ${fmt(sample.seconds)} s. Au-dessus du repos : ${net < 0 ? 'sous le bruit de fond' : fmt(net) + ' mJ/décision'}.${key === 'stick' ? ' Stick exclu.' : ''}`;
    const row = document.createElement('tr');
    for (const text of [names[key], ...['cpu', 'gpu', 'ane'].map(k => sample.components_w[k] === undefined ? '—' : fmt(sample.components_w[k], 2) + ' W'), fmt(sample.seconds) + ' s', String(sample.inferences)]) {
      const cell = document.createElement('td'); cell.textContent = text; row.append(cell);
    }
    rows.push(row);
  }
  $('energy-rows').replaceChildren(...rows);
  $('energy-breakdown').hidden = false;
  $('energy-note').textContent = 'Le repos est mesuré séparément juste avant chaque lot. Les mJ affichés sont bruts, repos inclus ; le détail donne le supplément estimé. Autres applications incluses. Le stick et le total électrique du Mac ne sont pas mesurés.';
}

function clearResults() {
  state.comparison = null; state.energy = null;
  $('token-count').textContent = 'Contexte max. 960 tokens, intentions incluses';
  for (const key of keys) {
    $('latency-' + key).textContent = '—';
    $('decision-' + key).textContent = 'En attente d’un message';
    $('confidence-' + key).textContent = '—'; $('bar-' + key).style.width = '0%';
    const unit = document.createElement('small'); unit.textContent = 'mJ / décision';
    $('energy-' + key).replaceChildren(document.createTextNode('— '), unit);
    $('energy-detail-' + key).textContent = 'Lancer une mesure d’énergie.';
  }
  $('energy-breakdown').hidden = true;
  const row = document.createElement('tr'), cell = document.createElement('td');
  cell.colSpan = 4; cell.className = 'empty'; cell.textContent = 'Lancer une comparaison pour ce message.';
  row.append(cell); $('probabilities').replaceChildren(row);
  $('comparison-note').textContent = 'Nouveau message : relancer les mesures. Les résultats précédents ont été effacés.';
  $('energy-note').textContent = 'Les valeurs incluent l’activité des autres applications. Il s’agit d’estimations de compteurs matériels, pas d’une mesure au wattmètre.';
  buttons();
}

async function run(kind) {
  state.busy = true; buttons(); $('error').hidden = true;
  try {
    const job = await request('/api/run', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({kind, message: $('message').value})});
    for (;;) {
      const current = await request('/api/jobs/' + job.id);
      $('status').textContent = current.progress;
      if (current.state === 'error') throw new Error(current.error);
      if (current.state === 'complete') {
        if (kind === 'compare') showComparison(current.result);
        if (kind === 'energy') showEnergy(current.result);
        $('status').textContent = kind === 'energy' ? 'Mesure énergétique terminée.' : 'Comparaison terminée.';
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 350));
    }
  } catch (error) {
    $('error').textContent = error.message; $('error').hidden = false;
    $('status').textContent = 'La requête n’a pas abouti.';
  } finally { state.busy = false; buttons(); }
}

$('compare').addEventListener('click', () => run('compare'));
$('energy').addEventListener('click', () => run('energy'));
$('reconnect').addEventListener('click', async () => { await run('reconnect'); await refreshStatus(); });
$('all-labels').addEventListener('change', renderProbabilities);
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => { $('message').value = button.dataset.example; clearResults(); $('message').focus(); }));
$('message').addEventListener('input', clearResults);
$('message').addEventListener('keydown', event => { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter' && !$('compare').disabled) { event.preventDefault(); run('compare'); } });
$('export').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({comparison: state.comparison, energy: state.energy}, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = 'crivo-local-results.json';
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
});
refreshStatus();
setInterval(() => { if (!state.busy) refreshStatus(); }, 2000);
