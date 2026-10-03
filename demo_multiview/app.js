const el = id => document.getElementById(id);
const state = {
  datasets: [], models: [], dataset: null, frameIndex: 0, modelId: null,
  payload: null, selected: null, timer: null, serial: 0, playing: false
};

function showMessage(text) {
  el('message').textContent = text;
  el('message').hidden = !text;
}

function stopPlayback() {
  if (state.timer) clearTimeout(state.timer);
  state.timer = null;
  state.playing = false;
  el('play').textContent = '▶ Play';
}

function currentFrame() { return state.dataset?.frames[state.frameIndex]; }

function setDataset(key) {
  stopPlayback();
  state.dataset = state.datasets.find(item => item.key === key);
  state.frameIndex = 0;
  state.selected = null;
  state.payload = null;
  if (!state.dataset) return;

  const methods = [...new Set(state.models.filter(model => model.dataset === key).map(model => model.method))];
  el('method').replaceChildren();
  for (const method of methods) {
    const option = document.createElement('option');
    option.value = method;
    option.textContent = method;
    el('method').append(option);
  }
  const annotations = document.createElement('option');
  annotations.value = 'annotations';
  annotations.textContent = 'Ground-truth annotations';
  el('method').append(annotations);

  selectMethod(methods[0] || 'annotations', false);
  const preferredFrame = state.models.find(model => model.id === state.modelId)?.default_frame;
  const preferredIndex = state.dataset.frames.indexOf(preferredFrame);
  if (preferredIndex >= 0) state.frameIndex = preferredIndex;
  el('frame-range').max = Math.max(0, state.dataset.frames.length - 1);
  loadFrame();
}

function selectMethod(method, reload = true) {
  stopPlayback();
  el('method').value = method;
  const choices = state.models.filter(model => model.dataset === state.dataset?.key && model.method === method);
  el('weight').replaceChildren();
  for (const model of choices) {
    const option = document.createElement('option');
    option.value = model.id;
    option.textContent = model.name;
    el('weight').append(option);
  }
  if (!choices.length) {
    const option = document.createElement('option');
    option.textContent = 'No checkpoint';
    el('weight').append(option);
  }
  state.modelId = choices[0]?.id || null;
  el('weight').disabled = !choices.length;
  el('run').textContent = choices.length ? 'Run inference' : 'Show annotations';
  state.selected = null;
  state.payload = null;
  if (reload) loadFrame();
}

async function loadFrame() {
  if (!state.dataset || !state.dataset.frames.length) {
    el('workspace').hidden = true;
    showMessage('No synchronized frames were found. Check the dataset path and camera image folders.');
    return;
  }
  const serial = ++state.serial;
  const frame = currentFrame();
  if (state.modelId && state.payload?.frame !== frame) state.selected = null;
  el('frame-label').textContent = String(frame);
  el('frame-range').value = state.frameIndex;
  el('prev').disabled = state.frameIndex === 0;
  el('next').disabled = state.frameIndex === state.dataset.frames.length - 1;
  el('run').disabled = true;
  el('model-status').textContent = state.modelId
    ? `Running ${el('method').value} on frame ${frame}...`
    : 'Loading annotations...';
  state.payload = { dataset: state.dataset.key, frame, source: 'pending', nodes: [], cameras: state.dataset.cameras };
  render();
  el('workspace').hidden = false;

  try {
    const url = state.modelId
      ? `/api/infer/${state.modelId}/${state.dataset.key}/${frame}`
      : `/api/frame/${state.dataset.key}/${frame}?source=annotations`;
    const response = await fetch(url);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Could not load this frame.');
    if (serial !== state.serial) return;
    state.payload = payload;
    if (!payload.nodes.some(node => node.id === state.selected)) state.selected = null;
    render();
    showMessage('');
    el('model-status').textContent = state.modelId
      ? `${el('method').value} · ${el('weight').selectedOptions[0]?.textContent} · ${payload.nodes.length} detections`
      : `Ground-truth annotations · ${payload.nodes.length} people`;
  } catch (error) {
    if (serial !== state.serial) return;
    showMessage(error.message);
    el('model-status').textContent = state.modelId ? 'Inference failed' : 'Could not load annotations';
    stopPlayback();
  } finally {
    if (serial === state.serial) el('run').disabled = false;
  }
}

function selectedNode() { return state.payload?.nodes.find(node => node.id === state.selected); }

function render() {
  const { dataset, payload } = state;
  el('camera-count').textContent = `${payload.cameras} cameras`;
  el('node-count').textContent = payload.source === 'pending' ? 'Loading...' : `${payload.nodes.length} nodes`;
  el('heatmap').hidden = !payload.heatmap;
  if (payload.heatmap) el('heatmap').src = payload.heatmap;
  document.querySelector('.map-axis-x').textContent = dataset.key === 'wildtrack' ? 'Y / WORLD GRID' : 'X / WORLD GRID';
  document.querySelector('.map-axis-y').textContent = dataset.key === 'wildtrack' ? 'X / WORLD GRID' : 'Y / WORLD GRID';
  el('source-note').textContent = state.modelId
    ? 'Ground-plane points come from the selected checkpoint. Detection IDs are frame-local.'
    : 'Points, boxes, and person IDs come from dataset annotations.';
  renderMap();
  renderCameras();
  renderDetail();
}

function renderMap() {
  const container = el('map-nodes');
  container.replaceChildren();
  state.payload.nodes.forEach(node => {
    const button = document.createElement('button');
    button.className = 'node' + (node.id === state.selected ? ' selected' : '');
    button.style.left = `${Math.min(100, Math.max(0, node.point[0] * 100))}%`;
    button.style.top = `${Math.min(100, Math.max(0, node.point[1] * 100))}%`;
    button.title = `ID ${node.id} · projected box in ${node.boxes.length} cameras`;
    button.setAttribute('aria-label', button.title);
    button.setAttribute('aria-pressed', String(node.id === state.selected));
    if (node.id === state.selected) {
      const label = document.createElement('span');
      label.className = 'node-label';
      label.textContent = node.id;
      button.append(label);
    }
    button.addEventListener('click', () => { state.selected = node.id; render(); });
    container.append(button);
  });
}

function renderCameras() {
  const container = el('cameras');
  container.replaceChildren();
  const node = selectedNode();
  for (let camera = 0; camera < state.payload.cameras; camera++) {
    const box = node?.boxes.find(item => item.camera === camera);
    const card = document.createElement('article');
    card.className = 'camera-card' + (box ? ' active' : '');
    const head = document.createElement('div');
    head.className = 'camera-head';
    const name = document.createElement('strong');
    name.textContent = `cam_${camera}`;
    const status = document.createElement('span');
    status.textContent = node && !box ? 'No projected box' : '';
    head.append(name, status);
    const wrap = document.createElement('div');
    wrap.className = 'image-wrap';
    const img = document.createElement('img');
    img.src = `/api/image/${state.dataset.key}/${state.payload.frame}/${camera}`;
    img.alt = `Camera ${camera + 1}, frame ${state.payload.frame}`;
    img.loading = camera > 3 ? 'lazy' : 'eager';
    wrap.append(img);
    if (box) {
      const overlay = document.createElement('div');
      overlay.className = 'box-layer';
      const rect = document.createElement('div');
      rect.className = 'bbox';
      const [x1, y1, x2, y2] = box.bbox;
      const draw = () => {
        const width = img.naturalWidth || 1920;
        const height = img.naturalHeight || 1080;
        rect.style.left = `${Math.max(0, x1 / width * 100)}%`;
        rect.style.top = `${Math.max(0, y1 / height * 100)}%`;
        rect.style.width = `${Math.max(0, Math.min(width, x2) - Math.max(0, x1)) / width * 100}%`;
        rect.style.height = `${Math.max(0, Math.min(height, y2) - Math.max(0, y1)) / height * 100}%`;
        rect.classList.toggle('top', y1 / height < .09);
      };
      img.addEventListener('load', draw, { once: true });
      draw();
      const label = document.createElement('span');
      label.className = 'bbox-label';
      label.textContent = `ID ${node.id}`;
      rect.append(label);
      overlay.append(rect);
      wrap.append(overlay);
    }
    card.append(head, wrap);
    container.append(card);
  }
}

function renderDetail() {
  const detail = el('detail');
  detail.replaceChildren();
  const node = selectedNode();
  if (!node) {
    detail.className = 'detail empty';
    detail.textContent = state.payload.source === 'pending'
      ? 'Images are ready. The occupancy map is being generated.'
      : state.payload.nodes.length
        ? 'Click a ground-plane node to highlight its box in every camera view.'
        : 'No nodes were found for this frame.';
    return;
  }
  detail.className = 'detail';
  const top = document.createElement('div');
  top.className = 'detail-top';
  const idGroup = document.createElement('div');
  const title = document.createElement('p');
  title.className = 'detail-id';
  title.textContent = `ID ${node.id}`;
  const sub = document.createElement('p');
  sub.className = 'detail-sub';
  sub.textContent = `Ground grid (${Math.round(node.x)}, ${Math.round(node.y)}) · Position ${node.position}${node.score == null ? '' : ` · Score ${node.score.toFixed(3)}`}`;
  idGroup.append(title, sub);
  const badge = document.createElement('span');
  badge.className = 'detail-badge';
  badge.textContent = `${node.boxes.length}/${state.payload.cameras} camera views`;
  top.append(idGroup, badge);
  detail.append(top);
  const list = document.createElement('div');
  list.className = 'detail-list';
  for (let camera = 0; camera < state.payload.cameras; camera++) {
    const box = node.boxes.find(item => item.camera === camera);
    const row = document.createElement('div');
    row.className = 'detail-row';
    const name = document.createElement('span');
    name.textContent = `cam_${camera}`;
    const coordinates = document.createElement('span');
    coordinates.textContent = box ? box.bbox.map(Math.round).join(', ') : 'Not visible';
    row.append(name, coordinates);
    list.append(row);
  }
  detail.append(list);
  const note = document.createElement('p');
  note.className = 'detail-note';
  note.textContent = state.modelId
    ? 'Boxes are reference projections from rectangles.pom. This checkpoint does not produce 2D boxes or tracking IDs.'
    : 'Boxes and person IDs come directly from dataset annotations.';
  detail.append(note);
}

async function init() {
  try {
    const [datasetsResponse, modelsResponse] = await Promise.all([fetch('/api/datasets'), fetch('/api/models')]);
    if (!datasetsResponse.ok || !modelsResponse.ok) throw new Error('The demo API is unavailable.');
    state.datasets = await datasetsResponse.json();
    state.models = await modelsResponse.json();
    if (!state.datasets.length) {
      showMessage('No dataset is configured. Start the server with --wildtrack or --multiviewx.');
      return;
    }
    state.datasets.forEach(dataset => {
      const option = document.createElement('option');
      option.value = dataset.key;
      option.textContent = dataset.name;
      el('dataset').append(option);
    });
    setDataset(state.datasets[0].key);
  } catch (error) {
    showMessage(`Could not connect to the demo server: ${error.message}`);
  }
}

el('dataset').addEventListener('change', event => setDataset(event.target.value));
el('method').addEventListener('change', event => selectMethod(event.target.value));
el('weight').addEventListener('change', event => {
  state.modelId = event.target.value;
  state.selected = null;
  state.payload = null;
  loadFrame();
});
el('run').addEventListener('click', () => { state.selected = null; loadFrame(); });
el('prev').addEventListener('click', () => {
  if (state.dataset && state.frameIndex > 0) { state.frameIndex--; loadFrame(); }
});
el('next').addEventListener('click', () => {
  if (state.dataset && state.frameIndex < state.dataset.frames.length - 1) { state.frameIndex++; loadFrame(); }
});
el('frame-range').addEventListener('input', event => {
  state.frameIndex = Number(event.target.value);
  loadFrame();
});
el('play').addEventListener('click', () => {
  if (!state.dataset?.frames.length) return;
  if (state.playing) { stopPlayback(); return; }
  state.playing = true;
  el('play').textContent = 'Ⅱ Pause';
  const advance = async () => {
    if (!state.playing) return;
    state.frameIndex = state.frameIndex >= state.dataset.frames.length - 1 ? 0 : state.frameIndex + 1;
    await loadFrame();
    if (state.playing) state.timer = setTimeout(advance, 650);
  };
  state.timer = setTimeout(advance, 650);
});
window.addEventListener('keydown', event => {
  if (event.target.closest('input,select,button')) return;
  if (event.key === 'ArrowLeft') el('prev').click();
  if (event.key === 'ArrowRight') el('next').click();
});
init();
