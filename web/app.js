const state = { history: null, audio: null, urls: [], requiresHistory: true };
const $ = (id) => document.getElementById(id);

async function connect() {
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    $('status').classList.add('ready');
    $('status').querySelector('span').textContent = `${data.checkpoint} · ${data.sample_rate} Hz`;
    $('variant').textContent = data.variant;
    $('device').textContent = data.device.toUpperCase();
    state.requiresHistory = data.requires_history;
    if (!state.requiresHistory) {
      $('historyHelp').textContent = '当前最佳真实录音模型不需要历史 TOL；此项可留空。';
      $('historyMeta').textContent = '可选 · 仅供切换预测条件模型时使用';
      $('subtitle').textContent = '真实连续海洋录音验证的 BS-RoFormer，本机完成分离。';
      $('forecastCard').classList.add('hidden');
    }
    ready();
  } catch (_) {
    $('status').querySelector('span').textContent = '模型未连接';
  }
}

function ready() { $('runButton').disabled = !(state.audio && (!state.requiresHistory || state.history)); }

$('historyFile').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  state.history = await file.text();
  const rows = state.history.trim().split(/\r?\n/).length - 1;
  $('historyName').textContent = file.name;
  $('historyMeta').textContent = `${rows} 行历史记录 · ${(file.size / 1024).toFixed(1)} KB`;
  ready();
});

$('audioFile').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;
  state.audio = file;
  $('audioName').textContent = file.name;
  $('audioMeta').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · 本机处理`;
  ready();
});

function fileBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function audioUrl(base64) {
  const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
  const url = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
  state.urls.push(url);
  return url;
}

function drawSpectrum(values) {
  const canvas = $('spectrum');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = 'rgba(171,211,202,.12)'; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) { const y = height * i / 4; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(width,y); ctx.stroke(); }
  if (!values?.length) return;
  const min = Math.min(...values), max = Math.max(...values), range = Math.max(1, max - min);
  const gradient = ctx.createLinearGradient(0, 0, width, 0); gradient.addColorStop(0, '#6ee7c8'); gradient.addColorStop(1, '#f0bd68');
  ctx.strokeStyle = gradient; ctx.lineWidth = 2; ctx.beginPath();
  values.forEach((value, index) => {
    const x = index * width / (values.length - 1); const y = height - 10 - (value - min) / range * (height - 20);
    index ? ctx.lineTo(x,y) : ctx.moveTo(x,y);
  });
  ctx.stroke();
}

$('runButton').addEventListener('click', async () => {
  $('error').classList.add('hidden'); $('results').classList.add('hidden'); $('progressPanel').classList.remove('hidden');
  $('runButton').disabled = true;
  try {
    const audio_base64 = await fileBase64(state.audio);
    const response = await fetch('/api/separate', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({audio_base64, history_csv: state.history}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '分离失败');
    state.urls.forEach(URL.revokeObjectURL); state.urls = [];
    const source = audioUrl(data.source_wav), ocean = audioUrl(data.ocean_wav);
    $('sourcePlayer').src = source; $('sourceDownload').href = source;
    $('oceanPlayer').src = ocean; $('oceanDownload').href = ocean;
    $('durationMetric').textContent = `${data.duration_seconds.toFixed(2)} s`;
    $('elapsedMetric').textContent = `${data.elapsed_seconds.toFixed(2)} s`;
    $('rtfMetric').textContent = data.real_time_factor.toFixed(3);
    drawSpectrum(data.forecast_db);
    $('results').classList.remove('hidden');
  } catch (error) {
    $('error').textContent = error.message; $('error').classList.remove('hidden');
  } finally {
    $('progressPanel').classList.add('hidden'); ready();
  }
});

connect();
