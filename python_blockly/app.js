/* Blockly creates source text; /run executes it using real local CPython. */
let workspace;
async function startEditor() {
  const response = await fetch('/api/schema');
  const schema = await response.json();
  if (!response.ok) throw new Error(schema.error || 'Could not discover the VSSS API.');
  installVsssBlocks(schema);
  workspace = Blockly.inject('blocklyDiv', {
  media: 'node_modules/blockly/media/',
  toolbox: {kind: 'categoryToolbox', contents: [
    ...vsssToolbox,
    {kind: 'category', name: 'Text', colour: 160, contents: [
      {kind: 'block', type: 'text_print', inputs: {TEXT: {shadow: {type: 'text', fields: {TEXT: 'Hello from Python!'}}}}},
      {kind: 'block', type: 'text'}]},
    {kind: 'category', name: 'Loops', colour: 120, contents: [
      {kind: 'block', type: 'vsss_wait', inputs: {SECONDS: {shadow: {type: 'math_number', fields: {NUM: 1}}}}},
      {kind: 'block', type: 'controls_repeat_ext', inputs: {TIMES: {shadow: {type: 'math_number', fields: {NUM: 3}}}}},
      {kind: 'block', type: 'controls_whileUntil'}, {kind: 'block', type: 'controls_forEach'},
      {kind: 'block', type: 'controls_flow_statements'}]},
    {kind: 'category', name: 'Math', colour: 230, contents: [
      {kind: 'block', type: 'math_number'},
      {kind: 'block', type: 'math_arithmetic', fields: {OP: 'ADD'}},
      {kind: 'block', type: 'math_constrain'}]},
    {kind: 'category', name: 'Logic', colour: 210, contents: [
      {kind: 'block', type: 'controls_if'}, {kind: 'block', type: 'logic_compare'},
      {kind: 'block', type: 'logic_boolean'}, {kind: 'block', type: 'logic_null'},
      {kind: 'block', type: 'logic_operation'}, {kind: 'block', type: 'logic_negate'}]},
    {kind: 'category', name: 'Lists', colour: 260, contents: [
      {kind: 'block', type: 'vsss_index'},
      {kind: 'block', type: 'lists_create_with'}, {kind: 'block', type: 'lists_length'},
      {kind: 'block', type: 'lists_isEmpty'}]},
    {kind: 'category', name: 'Variables', colour: 330, custom: 'VARIABLE'}
  ]},
  trashcan: true, grid: {spacing: 24, length: 2, colour: '#d6deeb', snap: true},
  zoom: {controls: true, wheel: true, startScale: 0.95}
});
  workspace.addChangeListener(updateCode);
  loadLesson('printing');
}

function generatedCode() {
  const raw = python.pythonGenerator.workspaceToCode(workspace).trim() || 'pass';
  const types = workspace.getAllBlocks(false).map(b => b.type);
  const carTypes = ['vsss_connect','vsss_car','vsss_velocity','vsss_waypoint','vsss_stop','vsss_wait_pose','vsss_pose','vsss_position','vsss_visible','vsss_status','vsss_target_reached','vsss_onboard_ready','vsss_robot_fault','vsss_wait_arrival'];
  const imports = [];
  if(types.some(t => t.startsWith('api_car_') || carTypes.includes(t))) imports.push('Car');
  if(types.some(t => t.startsWith('api_gps_'))) imports.push('gps');
  if(types.some(t => t.startsWith('api_input_') || ['vsss_key','vsss_key_held','vsss_pad_held','vsss_pad_axis','vsss_pad_button'].includes(t))) imports.push('input');
  const times = ['sleep','monotonic'].filter(name => new RegExp('\\b' + name + '\\(').test(raw));
  let header = times.length ? `from time import ${times.join(', ')}\n` : '';
  if(!imports.length) return header + (header ? '\n' : '') + raw + '\n';
  header = `from vsss import ${[...imports,'run'].join(', ')}\n` + header;
  return header + '\n\n@run\ndef main():\n' + raw.split('\n').map(line=>'    '+line).join('\n') + '\n\n\nif __name__ == "__main__":\n    main()\n';
}

let activeRun = null;
let busy = false;
function updateCode() {
  try { document.getElementById('code').textContent = generatedCode() || '# Connect some blocks to begin.'; }
  catch (error) { document.getElementById('code').textContent = `# ${error.message}`; }
  try {
    checkCarBlocks(workspace);
    if (!busy) document.getElementById('status').textContent = 'Ready';
  } catch (error) { document.getElementById('status').textContent = error.message; }
}
document.getElementById('lesson').onchange = event => {if(event.target.value) loadLesson(event.target.value);};
async function request(path, payload = {}, options = {}) {
  const response = await fetch(path, {method: 'POST', headers: {
    'Content-Type': 'application/json', 'X-VSSS-Blockly': '1'
  }, body: JSON.stringify(payload), ...options});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Server returned ${response.status}`);
  return result;
}
function setBusy(value) {
  busy = value;
  for (const id of ['run', 'load', 'lesson', 'environment', 'reset-environment']) document.getElementById(id).disabled = value;
}
function showRun(result) {
  document.getElementById('output').textContent = result.output + (result.error ? '\n' + result.error : '') || '(No output yet.)';
  const labels = {
    running: 'Running', stopping: 'Stopping and disconnecting…',
    finished: 'Finished — car stopped and disconnected', stopped: result.reason || 'Stopped',
    failed: 'Run failed — see output', idle: 'No program running'
  };
  document.getElementById('status').textContent = labels[result.state] || result.state;
  const active = ['running', 'stopping'].includes(result.state);
  setBusy(active);
  if (!active) activeRun = null;
}
async function pollRun(runId = activeRun) {
  if (!runId || activeRun !== runId) return;
  try {
    const result = await request('/status', {run_id: runId});
    if (activeRun !== runId) return;
    showRun(result);
  } catch (error) {
    if (activeRun === runId) document.getElementById('status').textContent = 'Connection lost; the server stops runs after 5 seconds without a heartbeat.';
  }
  if (activeRun === runId) setTimeout(() => pollRun(runId), 500);
}
document.getElementById('run').onclick = async () => {
  setBusy(true);
  document.getElementById('status').textContent = 'Starting program…';
  document.getElementById('output').textContent = 'Starting…';
  try {
    checkCarBlocks(workspace);
    const result = await request('/run', {code: generatedCode()});
    activeRun = result.run_id;
    showRun(result);
    if (activeRun) pollRun();
  } catch (error) {
    document.getElementById('status').textContent = 'Could not start';
    document.getElementById('output').textContent = error.message;
    setBusy(false);
  }
};
document.getElementById('stop').onclick = async () => {
  document.getElementById('status').textContent = 'Requesting stop…';
  try {
    const result = await request('/stop', activeRun ? {run_id: activeRun} : {});
    const wasPolling = Boolean(activeRun);
    activeRun = result.run_id || null;
    showRun(result);
    if (!wasPolling && activeRun) pollRun();
  } catch (error) { document.getElementById('status').textContent = `Stop request failed: ${error.message}`; }
};
window.addEventListener('pagehide', () => {
  if (activeRun) request('/stop', {run_id: activeRun}, {keepalive: true}).catch(() => {});
});
document.getElementById('save').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(Blockly.serialization.workspaces.save(workspace), null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'my-blocks.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
document.getElementById('load').onclick = () => document.getElementById('file').click();
document.getElementById('file').onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    if (file.size > 200000) throw new Error('File is too large.');
    const saved = JSON.parse(await file.text());
    Blockly.serialization.workspaces.load(saved, workspace);
    updateCode();
    document.getElementById('status').textContent = 'Blocks opened';
  } catch (error) { document.getElementById('status').textContent = `Could not open blocks: ${error.message}`; }
  event.target.value = '';
};
document.getElementById('export').onclick = () => {
  try {
    checkCarBlocks(workspace);
    const code = '# Generated by VSSS Blocks. Running this file uses real hardware.\n' +
      '# Check the car number and field before running in the lab.\n' + generatedCode();
    const url = URL.createObjectURL(new Blob([code], {type: 'text/x-python'}));
    const link = document.createElement('a'); link.href = url; link.download = 'student_program.py'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    document.getElementById('status').textContent = 'Python exported — running it separately can move a real robot';
  } catch (error) { document.getElementById('status').textContent = error.message; }
};
startEditor().catch(error => {
  document.getElementById('status').textContent = 'Editor could not start';
  document.getElementById('output').textContent = error.message;
  document.getElementById('run').disabled = true;
});

// The Python server renders the same plot as vsss-plot; canvas only displays it.
const plotCanvas = document.getElementById('field');
let plotState = null;
let dragPoint = null;
let dragState = null;
let pointerQueue = Promise.resolve();
const plotOwner = crypto.randomUUID();
function sendPointer(operation, point = dragPoint, state = dragState) {
  if (!point || !state) return;
  const payload = {operation, ...point, owner: plotOwner,
    source_id: state.source_id, generation: state.generation};
  pointerQueue = pointerQueue.then(() => request('/plot/pointer', payload)).catch(error => {
    document.getElementById('plot-status').textContent = error.message;
    dragPoint = dragState = null;
  });
}
function fieldPoint(event) {
  const rect = plotCanvas.getBoundingClientRect();
  return {x: (event.clientX-rect.left)*plotCanvas.width/rect.width,
          y: (event.clientY-rect.top)*plotCanvas.height/rect.height};
}
plotCanvas.onpointerdown = event => {
  if (event.button !== 0 || !plotState?.editable) return;
  event.preventDefault();
  plotCanvas.setPointerCapture(event.pointerId);
  dragPoint = fieldPoint(event); dragState = {...plotState};
  sendPointer('down');
};
plotCanvas.onpointermove = event => { if (dragPoint) dragPoint = fieldPoint(event); };
function releasePointer(event) {
  if (dragPoint) sendPointer('up', event ? fieldPoint(event) : dragPoint);
  dragPoint = dragState = null;
}
plotCanvas.onpointerup = releasePointer;
plotCanvas.onpointercancel = () => releasePointer();
plotCanvas.onlostpointercapture = () => releasePointer();
window.addEventListener('blur', () => releasePointer());
setInterval(() => { if (dragPoint) sendPointer('move'); }, 100);

async function pollPlot() {
  try {
    const response = await fetch('/api/plot', {cache: 'no-store'});
    if (response.status === 404) return; // Explicit fake-device test server.
    if (!response.ok) throw new Error('Plot unavailable');
    const metadata = JSON.parse(response.headers.get('X-VSSS-State'));
    const bitmap = await createImageBitmap(await response.blob());
    if (dragState && (dragState.source_id !== metadata.source_id || dragState.generation !== metadata.generation)) releasePointer();
    plotCanvas.width = bitmap.width; plotCanvas.height = bitmap.height;
    plotCanvas.getContext('2d').drawImage(bitmap, 0, 0); bitmap.close();
    plotState = metadata;
    plotCanvas.classList.toggle('editable', metadata.editable);
    document.getElementById('plot-status').textContent = metadata.status +
      (metadata.editable ? ' · Drag centres/balls to move; edges to rotate.' : ' · Observation only');
  } catch (error) { document.getElementById('plot-status').textContent = error.message; }
  setTimeout(pollPlot, 80);
}
let currentEnvironment = 'goalie_easy';
function environmentLabel(result) {
  currentEnvironment = result.selected;
  const live = result.selected === 'live';
  document.getElementById('mode-badge').textContent = live ? 'LIVE ROBOTS' : 'SIMULATION';
  document.getElementById('run').textContent = live ? '▶ Run live' : '▶ Run';
  document.getElementById('reset-environment').textContent = live ? 'Connect live' : 'Start / reset scenario';
  document.getElementById('run-hint').textContent = live ?
    'Live controls real cars. Check car numbers and clear the field.' : 'Run controls the selected simulator scenario.';
  document.getElementById('environment-status').textContent = result.mode ?
    `${result.selected} · ${result.mode}` : 'Choose a scenario, or press Run to start.';
}
async function chooseEnvironment() {
  releasePointer();
  setBusy(true);
  document.getElementById('environment-status').textContent = 'Opening environment…';
  try {
    const result = await request('/environment', {selection: document.getElementById('environment').value});
    environmentLabel(result);
  } catch (error) {
    document.getElementById('environment').value = currentEnvironment;
    document.getElementById('environment-status').textContent = error.message;
  }
  finally { setBusy(false); }
}
document.getElementById('environment').onchange = chooseEnvironment;
document.getElementById('reset-environment').onclick = chooseEnvironment;
fetch('/api/environment').then(response => response.json()).then(result => {
  const select = document.getElementById('environment');
  for (const item of result.options) select.add(new Option(item.label, item.id));
  if (result.options.length) { select.value = result.selected; environmentLabel(result); pollPlot(); }
}).catch(error => { document.getElementById('environment-status').textContent = error.message; });
