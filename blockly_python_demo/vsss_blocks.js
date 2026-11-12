/* Student blocks generate the existing public Car API, without changing it. */
Blockly.defineBlocksWithJsonArray([
  {type: 'vsss_car', message0: 'with car %1 %2 do %3', args0: [
    {type: 'field_number', name: 'NUMBER', value: 1, min: 0, precision: 1},
    {type: 'input_dummy'}, {type: 'input_statement', name: 'DO'}
  ], colour: 25, tooltip: 'Connect to carN. Always stop and disconnect on exit, including errors.'},
  {type: 'vsss_velocity', message0: 'set velocity forward %1 left %2 turn CCW %3', args0: [
    {type: 'input_value', name: 'X', check: 'Number'},
    {type: 'input_value', name: 'Y', check: 'Number'},
    {type: 'input_value', name: 'W', check: 'Number'}
  ], previousStatement: null, nextStatement: null, colour: 25,
  tooltip: 'Robot-frame commands from -1 to 1. These are normalized commands, not metres per second. Motion continues until changed or stopped.'},
  {type: 'vsss_waypoint', message0: 'go to field X %1 m Y %2 m heading %3 degrees', args0: [
    {type: 'input_value', name: 'X', check: 'Number'},
    {type: 'input_value', name: 'Y', check: 'Number'},
    {type: 'input_value', name: 'ANGLE', check: 'Number'}
  ], previousStatement: null, nextStatement: null, colour: 25,
  tooltip: 'Start onboard waypoint control. Returns when the command is acknowledged, not when the car arrives. X: -0.8 to 0.8 m; Y: -1.5 to 1.5 m.'},
  {type: 'vsss_stop', message0: 'stop car', previousStatement: null, nextStatement: null, colour: 25},
  {type: 'vsss_wait', message0: 'wait %1 seconds', args0: [
    {type: 'input_value', name: 'SECONDS', check: 'Number'}
  ], previousStatement: null, nextStatement: null, colour: 25,
  tooltip: 'Let time pass. An active velocity or waypoint continues during the wait.'},
  {type: 'vsss_wait_pose', message0: 'wait for camera position (timeout %1 seconds)', args0: [
    {type: 'input_value', name: 'TIMEOUT', check: 'Number'}
  ], previousStatement: null, nextStatement: null, colour: 190,
  tooltip: 'Wait for fresh camera feedback, or raise a timeout error.'},
  {type: 'vsss_pose', message0: 'camera pose (X, Y, heading)', output: null, colour: 190,
  tooltip: 'car.pose: Pose in metres/degrees, or None if missing or stale.'},
  {type: 'vsss_position', message0: 'camera position (X, Y)', output: null, colour: 190,
  tooltip: 'car.position: (x, y) in metres, or None if missing or stale.'},
  {type: 'vsss_visible', message0: 'camera position available?', output: 'Boolean', colour: 190},
  {type: 'vsss_status', message0: 'robot status', output: null, colour: 190,
  tooltip: 'Read onboard mode, fault and telemetry status.'}
]);

const py = python.pythonGenerator;
py.addReservedWords('Car,car,sleep');
const value = (block, generator, name, fallback = '0') =>
  generator.valueToCode(block, name, python.Order.NONE) || fallback;
py.forBlock.vsss_car = (block, generator) => {
  generator.definitions_.vsss_car_import = 'from car import Car';
  return `with Car(${block.getFieldValue('NUMBER')}) as car:\n` +
    (generator.statementToCode(block, 'DO') || generator.INDENT + 'pass\n');
};
py.forBlock.vsss_velocity = (block, generator) =>
  `car.set_velocity(x=${value(block, generator, 'X')}, y=${value(block, generator, 'Y')}, w=${value(block, generator, 'W')})\n`;
py.forBlock.vsss_waypoint = (block, generator) =>
  `car.go_to(${value(block, generator, 'X')}, ${value(block, generator, 'Y')}, ${value(block, generator, 'ANGLE', '90')})\n`;
py.forBlock.vsss_stop = () => 'car.stop()\n';
py.forBlock.vsss_wait = (block, generator) => {
  generator.definitions_.vsss_sleep_import = 'from time import sleep';
  return `sleep(${value(block, generator, 'SECONDS', '1')})\n`;
};
py.forBlock.vsss_wait_pose = (block, generator) => `car.wait_for_position(timeout=${value(block, generator, 'TIMEOUT', '10')})\n`;
py.forBlock.vsss_pose = () => ['car.pose', python.Order.MEMBER];
py.forBlock.vsss_position = () => ['car.position', python.Order.MEMBER];
py.forBlock.vsss_visible = () => ['car.pose is not None', python.Order.RELATIONAL];
py.forBlock.vsss_status = () => ['car.read_status()', python.Order.FUNCTION_CALL];

function numberInput(number) { return {shadow: {type: 'math_number', fields: {NUM: number}}}; }
const vsssToolbox = [
  {kind: 'category', name: 'Car', colour: 25, contents: [
    {kind: 'block', type: 'vsss_car'},
    {kind: 'block', type: 'vsss_velocity', inputs: {X: numberInput(.2), Y: numberInput(0), W: numberInput(0)}},
    {kind: 'block', type: 'vsss_waypoint', inputs: {X: numberInput(0), Y: numberInput(.5), ANGLE: numberInput(90)}},
    {kind: 'block', type: 'vsss_wait', inputs: {SECONDS: numberInput(1)}},
    {kind: 'block', type: 'vsss_stop'}
  ]},
  {kind: 'category', name: 'Sensors', colour: 190, contents: [
    {kind: 'block', type: 'vsss_wait_pose', inputs: {TIMEOUT: numberInput(10)}},
    ...['vsss_pose', 'vsss_position', 'vsss_visible', 'vsss_status'].map(type => ({kind: 'block', type}))
  ]}
];

// Every car/sensor block must be inside one car session. Wait is also useful alone.
function checkCarBlocks(workspace) {
  const sessions = workspace.getBlocksByType('vsss_car', false);
  if (sessions.length > 1) throw new Error('Use one “with car” block per program.');
  for (const block of workspace.getAllBlocks(false)) {
    if (!block.type.startsWith('vsss_') || ['vsss_car', 'vsss_wait'].includes(block.type)) continue;
    let parent = block.getParent();
    while (parent && parent.type !== 'vsss_car') parent = parent.getParent();
    if (!parent) throw new Error('Put car and sensor blocks inside the “with car” block.');
  }
}

function loadCarExample() {
  const printPose = () => ({type: 'text_print', inputs: {TEXT: {block: {type: 'vsss_pose'}}}});
  const steps = [
    {type: 'vsss_wait_pose', inputs: {TIMEOUT: numberInput(10)}},
    printPose(),
    {type: 'vsss_velocity', inputs: {X: numberInput(.2), Y: numberInput(0), W: numberInput(0)}},
    {type: 'vsss_wait', inputs: {SECONDS: numberInput(1)}},
    {type: 'vsss_stop'}, printPose(),
    {type: 'vsss_waypoint', inputs: {X: numberInput(0), Y: numberInput(.5), ANGLE: numberInput(90)}},
    {type: 'vsss_wait', inputs: {SECONDS: numberInput(2)}},
    {type: 'vsss_stop'},
    {type: 'text_print', inputs: {TEXT: {block: {type: 'vsss_status'}}}}
  ];
  for (let i = 0; i < steps.length - 1; i++) steps[i].next = {block: steps[i + 1]};
  Blockly.serialization.workspaces.load({blocks: {languageVersion: 0, blocks: [
    {type: 'vsss_car', x: 25, y: 25, fields: {NUMBER: 1}, inputs: {DO: {block: steps[0]}}}
  ]}}, workspace);
  updateCode();
}
