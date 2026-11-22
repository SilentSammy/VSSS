/* The server approves API members and supplies current Python signatures. */
const py = python.pythonGenerator;
py.addReservedWords('Car,gps,run,input,sleep,monotonic,main,arrival_deadline,arrival_status');
let vsssToolbox = [];
let vsssSchema;
const numberInput = n => ({shadow: {type: 'math_number', fields: {NUM: n}}});
const varField = name => ({id: 'v-' + name});
const variable = name => ({type: 'variables_get', fields: {VAR: varField(name)}});
const keyboardLetters = ['w', 'a', 's', 'd', 'q', 'e',
  ...'abcdefghijklmnopqrstuvwxyz'.split('').filter(key => !'wasdqe'.includes(key))];
const keyboardKeyOptions = [
  ...keyboardLetters.map(key => [key.toUpperCase(), key]),
  ...'0123456789'.split('').map(key => [key, key]),
  ['Space', 'Key.space'], ['Enter', 'Key.enter'], ['Shift', 'Key.shift'],
  ['Arrow up', 'Key.up'], ['Arrow down', 'Key.down'],
  ['Arrow left', 'Key.left'], ['Arrow right', 'Key.right']
];
const carName = (block, generator) => generator.getVariableName(block.getFieldValue('CAR'));
const value = (block, generator, name, fallback = '0') => generator.valueToCode(block, name, python.Order.NONE) || fallback;
function literal(data) {
  if (data === null) return 'None';
  if (typeof data === 'boolean') return data ? 'True' : 'False';
  if (typeof data === 'number') {
    if (!Number.isFinite(data)) throw new Error('Data numbers must be finite.');
    return String(data);
  }
  if (typeof data === 'string') return JSON.stringify(data);
  if (Array.isArray(data)) return '[' + data.map(literal).join(', ') + ']';
  return '{' + Object.entries(data).map(([k, v]) => literal(k) + ': ' + literal(v)).join(', ') + '}';
}
function shadow(arg) {
  if (arg.type === 'Number') return numberInput(arg.example ?? 0);
  if (arg.type === 'Boolean') return {shadow: {type: 'logic_boolean', fields: {BOOL: arg.example ? 'TRUE' : 'FALSE'}}};
  if (arg.type === 'String' && arg.example !== null) return {shadow: {type: 'text', fields: {TEXT: arg.example ?? ''}}};
  return {shadow: {type: 'logic_null'}};
}
function installVsssBlocks(schema) {
  vsssSchema = schema;
  const definitions = [
    {type: 'vsss_connect', message0: 'use car %1 as %2', args0: [
      {type: 'input_value', name: 'NUMBER', check: 'Number'},
      {type: 'field_variable', name: 'CAR', variable: 'car1'}], previousStatement: null, nextStatement: null, colour: 25,
      tooltip: 'Connect over BLE only. Reuse this variable for commands. @run cleans up all cars.'},
    {type: 'vsss_wait', message0: 'wait %1 seconds', args0: [{type: 'input_value', name: 'SECONDS', check: 'Number'}], previousStatement: null, nextStatement: null, colour: 120},
    {type: 'vsss_monotonic', message0: 'elapsed clock seconds', output: 'Number', colour: 120},
    {type: 'vsss_field', message0: 'field %1 of %2', args0: [
      {type: 'field_dropdown', name: 'FIELD', options: schema.record_fields.map(f => [f, f])},
      {type: 'input_value', name: 'OBJECT'}], output: null, colour: 190,
      tooltip: 'Read a pose or status field. Check that a pose is not None before reading it.'},
    {type: 'vsss_index', message0: 'item with key / index %1 in %2', args0: [
      {type: 'input_value', name: 'INDEX'}, {type: 'input_value', name: 'OBJECT'}], output: null, colour: 260,
      tooltip: 'Dictionary key or zero-based list index.'},
    {type: 'vsss_raise', message0: 'fail with message %1', args0: [{type: 'input_value', name: 'TEXT', check: 'String'}], previousStatement: null, nextStatement: null, colour: 0},
    {type: 'vsss_pad_axis', message0: 'gamepad %1 axis %2', args0: [
      {type: 'field_number', name: 'PAD', value: 0, min: 0, max: 9, precision: 1},
      {type: 'field_dropdown', name: 'AXIS', options: ['LX','LY','RX','RY','LT','RT','DPAD_X','DPAD_Y'].map(x => [x, x])}], output: 'Number', colour: 290},
    {type: 'vsss_pad_button', message0: 'gamepad %1 button %2 %3', args0: [
      {type: 'field_number', name: 'PAD', value: 0, min: 0, max: 9, precision: 1},
      {type: 'field_dropdown', name: 'BUTTON', options: ['A','B','X','Y','LB','RB','LT','RT','L3','R3','START','SLCT','GUIDE','DPAD_UP','DPAD_DOWN','DPAD_LEFT','DPAD_RIGHT'].map(x => [x, x])},
      {type: 'field_dropdown', name: 'MODE', options: [['held','is_pressed'],['just pressed','rising_edge'],['just released','falling_edge'],['toggled','is_toggled']]}], output: 'Boolean', colour: 290},
    {type: 'vsss_key', message0: 'keyboard key %1 %2', args0: [
      {type: 'field_dropdown', name: 'KEY', options: keyboardKeyOptions},
      {type: 'field_dropdown', name: 'MODE', options: [['held','is_pressed'],['just pressed','rising_edge'],['just released','falling_edge'],['toggled','is_toggled']]}], output: 'Boolean', colour: 290,
      tooltip: 'Read a safe, commonly used keyboard key on the server computer.'},
    {type: 'vsss_key_held', message0: 'keyboard key %1 held?', args0: [{type:'field_input', name:'KEY', text:'w'}], output:'Boolean', colour:290},
    {type: 'vsss_pad_held', message0: 'gamepad %1 button %2 held?', args0: [
      {type:'field_number',name:'PAD',value:0,min:0,max:9,precision:1},
      {type:'field_dropdown',name:'BUTTON',options:['A','B','X','Y','LB','RB','LT','RT','L3','R3','START'].map(x=>[x,x])}],output:'Boolean',colour:290},
    ...[
      ['vsss_target_reached','target reached?'],['vsss_onboard_ready','onboard position available?'],['vsss_robot_fault','robot fault?']
    ].map(([type,label])=>({type,message0:'%1 '+label,args0:[{type:'field_variable',name:'CAR',variable:'car1'}],output:'Boolean',colour:190})),
    {type:'vsss_wait_arrival',message0:'wait until %1 reaches target (timeout %2 seconds)',args0:[
      {type:'field_variable',name:'CAR',variable:'car1'},{type:'input_value',name:'TIMEOUT',check:'Number'}],previousStatement:null,nextStatement:null,colour:190}
  ];
  const colours = {'Cars':25, 'Movement':25, 'Car feedback':190, 'Advanced lessons':210, 'Field':190, 'Input':290};
  const categories = new Map(Object.keys(colours).map(name => [name, {kind:'category', name, colour:colours[name], contents:[]} ]));
  for (const spec of schema.blocks) {
    const args0 = [];
    let message = spec.owner === 'car' ? '%1 ' + spec.label : spec.owner + ' ' + spec.label;
    if (spec.owner === 'car') args0.push({type:'field_variable', name:'CAR', variable:'car1'});
    for (const arg of spec.args) {
      let label = arg.name.replaceAll('_', ' ');
      if (spec.member === 'set_velocity') label = {x:'forward', y:'left', w:'turn CCW'}[arg.name] || label;
      if (spec.member === 'go_to') label = {x:'X (m)', y:'Y (m)', angle:'heading (degrees)'}[arg.name] || label;
      args0.push({type:'input_value', name:arg.name, ...(arg.type ? {check:arg.type} : {})});
      message += ` ${label} %${args0.length}`;
    }
    definitions.push({type:spec.id, message0:message, args0, colour:colours[spec.category], tooltip:spec.tooltip,
      ...(spec.returns ? {output:null} : {previousStatement:null,nextStatement:null})});
    py.forBlock[spec.id] = (block, generator) => {
      const target = spec.owner === 'car' ? carName(block, generator) : spec.owner;
      const params = spec.args.map(arg => `${arg.name}=${value(block,generator,arg.name,literal(arg.default))}`).join(', ');
      const code = target + '.' + spec.member + (spec.kind === 'method' ? '(' + params + ')' : '');
      return spec.returns ? [code, python.Order.MEMBER] : code + '\n';
    };
    categories.get(spec.category).contents.push({kind:'block', type:spec.id,
      inputs:Object.fromEntries(spec.args.map(arg => [arg.name,shadow(arg)]))});
  }
  Blockly.defineBlocksWithJsonArray(definitions);
  py.forBlock.vsss_connect = (b,g) => `${carName(b,g)} = Car(${value(b,g,'NUMBER','1')})\n`;
  py.forBlock.vsss_wait = (b,g) => `sleep(${value(b,g,'SECONDS','0.05')})\n`;
  py.forBlock.vsss_monotonic = () => ['monotonic()',python.Order.FUNCTION_CALL];
  py.forBlock.vsss_field = (b,g) => [`(${value(b,g,'OBJECT','None')}).${b.getFieldValue('FIELD')}`,python.Order.MEMBER];
  py.forBlock.vsss_index = (b,g) => [`(${value(b,g,'OBJECT','[]')})[${value(b,g,'INDEX')} ]`,python.Order.MEMBER];
  py.forBlock.vsss_raise = (b,g) => `raise RuntimeError(${value(b,g,'TEXT',JSON.stringify('Program stopped'))})\n`;
  py.forBlock.vsss_pad_axis = b => [`input.get_axis(${JSON.stringify(b.getFieldValue('AXIS')+b.getFieldValue('PAD'))})`,python.Order.FUNCTION_CALL];
  py.forBlock.vsss_pad_button = b => [`input.${b.getFieldValue('MODE')}(${JSON.stringify(b.getFieldValue('BUTTON')+b.getFieldValue('PAD'))})`,python.Order.FUNCTION_CALL];
  py.forBlock.vsss_key = b => [`input.${b.getFieldValue('MODE')}(${JSON.stringify(b.getFieldValue('KEY'))})`,python.Order.FUNCTION_CALL];
  py.forBlock.vsss_key_held = b => [`input.is_pressed(${JSON.stringify(b.getFieldValue('KEY'))})`,python.Order.FUNCTION_CALL];
  py.forBlock.vsss_pad_held = b => [`input.is_pressed(${JSON.stringify(b.getFieldValue('BUTTON')+b.getFieldValue('PAD'))})`,python.Order.FUNCTION_CALL];
  py.forBlock.vsss_target_reached = (b,g) => [`${carName(b,g)}.read_status().mode_name == 'reached'`,python.Order.RELATIONAL];
  py.forBlock.vsss_onboard_ready = (b,g) => [`${carName(b,g)}.read_status().pose_fresh`,python.Order.MEMBER];
  py.forBlock.vsss_robot_fault = (b,g) => [`${carName(b,g)}.read_status().fault_name != 'none'`,python.Order.RELATIONAL];
  py.forBlock.vsss_wait_arrival = (b,g) => {
    const car=carName(b,g);
    const timeout=value(b,g,'TIMEOUT','10');
    return `arrival_deadline = monotonic() + (${timeout})\nsleep(0.1)\nwhile True:\n    arrival_status = ${car}.read_status()\n    if arrival_status.fault_name != 'none' or not arrival_status.pose_fresh:\n        raise RuntimeError('Onboard position/control unavailable')\n    if arrival_status.mode_name == 'reached':\n        break\n    if monotonic() >= arrival_deadline:\n        raise TimeoutError('Waypoint timed out')\n    sleep(0.05)\n`;
  };
  categories.get('Cars').contents.unshift({kind:'block',type:'vsss_connect',inputs:{NUMBER:numberInput(1)}});
  categories.get('Input').contents.push(...['vsss_key','vsss_pad_button','vsss_pad_axis'].map(type=>({kind:'block',type})));
  categories.get('Car feedback').contents.push({kind:'block',type:'vsss_field'});
  categories.get('Advanced lessons').contents.push(
    ...['vsss_target_reached','vsss_onboard_ready','vsss_robot_fault','vsss_monotonic'].map(type=>({kind:'block',type})),
    {kind:'block',type:'vsss_wait_arrival',inputs:{TIMEOUT:numberInput(10)}},
    {kind:'block',type:'vsss_raise',inputs:{TEXT:{shadow:{type:'text',fields:{TEXT:'Program stopped'}}}}}
  );
  vsssToolbox = [...categories.values()].filter(c=>c.contents.length);
  installLegacyBlocks();
}

function checkCarBlocks(workspace) {
  for (const block of workspace.getAllBlocks(false)) {
    if (!['vsss_velocity','vsss_waypoint','vsss_stop','vsss_wait_pose','vsss_pose','vsss_position','vsss_visible','vsss_status'].includes(block.type)) continue;
    let parent = block.getParent();
    while (parent && parent.type !== 'vsss_car') parent = parent.getParent();
    if (!parent) throw new Error('Legacy car blocks must be inside their with-car block. Use the new named-car blocks for new programs.');
  }
}

function installLegacyBlocks() {
  const defs = [
    {type:'vsss_car',message0:'with car %1 %2 do %3',args0:[{type:'field_number',name:'NUMBER',value:1,min:0,precision:1},{type:'input_dummy'},{type:'input_statement',name:'DO'}],colour:25},
    {type:'vsss_velocity',message0:'set velocity forward %1 left %2 turn CCW %3',args0:['X','Y','W'].map(name=>({type:'input_value',name,check:'Number'})),previousStatement:null,nextStatement:null,colour:25},
    {type:'vsss_waypoint',message0:'go to field X %1 Y %2 heading %3',args0:['X','Y','ANGLE'].map(name=>({type:'input_value',name,check:'Number'})),previousStatement:null,nextStatement:null,colour:25},
    {type:'vsss_stop',message0:'stop car',previousStatement:null,nextStatement:null,colour:25},
    {type:'vsss_wait_pose',message0:'wait for camera position timeout %1',args0:[{type:'input_value',name:'TIMEOUT'}],previousStatement:null,nextStatement:null,colour:190},
    ...['pose','position','visible','status'].map(name=>({type:'vsss_'+name,message0:'legacy car '+name,output:null,colour:190}))
  ];
  Blockly.defineBlocksWithJsonArray(defs);
  py.forBlock.vsss_car=(b,g)=>`with Car(${b.getFieldValue('NUMBER')}) as car:\n`+(g.statementToCode(b,'DO')||g.INDENT+'pass\n');
  py.forBlock.vsss_velocity=(b,g)=>`car.set_velocity(x=${value(b,g,'X')}, y=${value(b,g,'Y')}, w=${value(b,g,'W')})\n`;
  py.forBlock.vsss_waypoint=(b,g)=>`car.go_to(${value(b,g,'X')}, ${value(b,g,'Y')}, ${value(b,g,'ANGLE','90')})\n`;
  py.forBlock.vsss_stop=()=> 'car.stop()\n';
  py.forBlock.vsss_wait_pose=(b,g)=>`car.wait_for_position(timeout=${value(b,g,'TIMEOUT','10')})\n`;
  py.forBlock.vsss_pose=()=>['car.pose',python.Order.MEMBER];
  py.forBlock.vsss_position=()=>['car.position',python.Order.MEMBER];
  py.forBlock.vsss_visible=()=>['car.pose is not None',python.Order.RELATIONAL];
  py.forBlock.vsss_status=()=>['car.read_status()',python.Order.FUNCTION_CALL];
}

function chain(steps) {for(let i=0;i<steps.length-1;i++) steps[i].next={block:steps[i+1]}; return steps[0];}
const expression = (type,fields={},inputs={}) => ({type,fields,inputs});
const inputBlock = block => ({block});
const num = n => inputBlock(expression('math_number',{NUM:n}));
const text = s => inputBlock(expression('text',{TEXT:s}));
const carBlock = (member,name='car1',inputs={}) => expression('api_car_'+member,{CAR:varField(name)},inputs);
const connect = (name,n) => expression('vsss_connect',{CAR:varField(name)},{NUMBER:num(n)});
const wait = n => expression('vsss_wait',{}, {SECONDS:num(n)});
const print = b => expression('text_print',{}, {TEXT:inputBlock(b)});
const field = (obj,name) => expression('vsss_field',{FIELD:name},{OBJECT:inputBlock(obj)});
const compare = (a,op,b) => expression('logic_compare',{OP:op},{A:inputBlock(a),B:inputBlock(b)});
const vsssLessonIds = new Set(['blank','printing','variables','rotation','kick','goalie','teleop',
  'read_gamepad','position','waypoints','observe','gamepads','keyboard','multi','velocity']);
function loadLesson(kind) {
  if (!vsssLessonIds.has(kind)) throw new Error('Unknown example: ' + kind + '. Refresh the editor.');
  let steps;
  document.getElementById('lesson').value = kind;
  if(kind==='blank') {
    workspace.clear();
    const variableMap = workspace.getVariableMap();
    for(const variable of variableMap.getAllVariables()) variableMap.deleteVariable(variable);
    updateCode();
    return;
  }
  const ids = ['car1','car2','pose','total','count','forward','turn','balls','ball','response'];
  const forever = steps => expression('controls_whileUntil',{MODE:'WHILE'},{BOOL:inputBlock(expression('logic_boolean',{BOOL:'TRUE'})),DO:inputBlock(chain(steps))});
  const assign = (name,block) => expression('variables_set',{VAR:varField(name)},{VALUE:inputBlock(block)});
  const firstBall = () => expression('vsss_index',{}, {INDEX:num(0),OBJECT:inputBlock(variable('balls'))});
  const ballCoordinate = name => field(variable('ball'),name);
  const ballCount = () => expression('lists_length',{}, {VALUE:inputBlock(variable('balls'))});
  if(kind==='printing') {
    steps=[print(expression('text',{TEXT:'Hello, Python!'})),print(expression('math_arithmetic',{OP:'ADD'},{A:num(7),B:num(5)}))];
  } else if(kind==='variables') {
    steps=[assign('total',expression('math_number',{NUM:0})),expression('controls_repeat_ext',{}, {TIMES:num(5),DO:inputBlock(chain([
      assign('total',expression('math_arithmetic',{OP:'ADD'},{A:inputBlock(variable('total')),B:num(2)})),print(variable('total'))
    ]))})];
  } else if(kind==='rotation') {
    steps=[connect('car1',1),forever([
      carBlock('set_velocity','car1',{x:num(0),y:num(0),w:num(.2)}),wait(1),
      carBlock('set_velocity','car1',{x:num(0),y:num(0),w:num(-.2)}),wait(1)
    ])];
  } else if(kind==='teleop') {
    steps=[connect('car1',1),forever([
      assign('forward',expression('api_input_get_bipolar_ctrl',{}, {high_key:text('w'),low_key:text('s'),keyboard_scale:num(.5)})),
      assign('turn',expression('math_arithmetic',{OP:'MULTIPLY'},{A:inputBlock(expression('vsss_pad_axis',{PAD:0,AXIS:'RX'})),B:num(.5)})),
      carBlock('set_velocity','car1',{x:inputBlock(variable('forward')),y:num(0),w:inputBlock(variable('turn'))}),wait(.05)
    ])];
  } else if(kind==='read_gamepad') {
    const readings=(label,blocks)=>print(expression('lists_create_with',{},Object.fromEntries(
      [expression('text',{TEXT:label}),...blocks].map((block,i)=>['ADD'+i,inputBlock(block)])
    )));
    const buttons=readings('Buttons A/B/X/Y',['A','B','X','Y'].map(BUTTON=>expression('vsss_pad_held',{PAD:0,BUTTON})));
    const sticks=readings('Sticks LX/LY/RX/RY',['LX','LY','RX','RY'].map(AXIS=>expression('vsss_pad_axis',{PAD:0,AXIS})));
    const triggers=readings('Triggers LT/RT',['LT','RT'].map(AXIS=>expression('vsss_pad_axis',{PAD:0,AXIS})));
    buttons.inputs.TEXT.block.extraState={itemCount:5};
    sticks.inputs.TEXT.block.extraState={itemCount:5};
    triggers.inputs.TEXT.block.extraState={itemCount:3};
    steps=[forever([buttons,sticks,triggers,wait(.25)])];
  } else if(kind==='position') {
    steps=[forever([assign('pose',expression('api_gps_get_car',{}, {number:num(1)})),print(variable('pose')),wait(.25)])];
  } else if(kind==='waypoints') {
    const arrival=()=>expression('vsss_wait_arrival',{CAR:varField('car1')},{TIMEOUT:num(10)});
    steps=[connect('car1',1),forever([
      carBlock('go_to','car1',{x:num(0),y:num(.5),angle:num(90)}),arrival(),
      carBlock('go_to','car1',{x:num(0),y:num(-.5),angle:num(90)}),arrival()
    ])];
  } else if(kind==='kick') {
    // One stationary ball, a clear approach, and a fixed +Y kick direction.
    steps=[connect('car1',1),expression('api_car_wait_for_position',{CAR:varField('car1')},{timeout:num(10)}),
      assign('balls',expression('api_gps_balls')),
      expression('controls_if',{}, {IF0:inputBlock(compare(ballCount(),'EQ',expression('math_number',{NUM:0}))),
        DO0:inputBlock(expression('vsss_raise',{}, {TEXT:text('One visible stationary ball is required.')}))}),
      assign('ball',firstBall()),
      print(expression('text',{TEXT:'Aligning 30 cm behind the ball, facing the top wall...'})),
      carBlock('go_to','car1',{x:inputBlock(ballCoordinate('x')),
        y:inputBlock(expression('math_arithmetic',{OP:'MINUS'},{A:inputBlock(ballCoordinate('y')),B:num(.3)})),angle:num(90)}),
      expression('vsss_wait_arrival',{CAR:varField('car1')},{TIMEOUT:num(10)}),
      print(expression('text',{TEXT:'Aligned. Kicking at full speed!'})),
      carBlock('set_velocity','car1',{x:num(1),y:num(0),w:num(0)}),wait(.8),carBlock('stop')];
  } else if(kind==='goalie') {
    // At heading 90, positive sideways velocity moves toward field -X.
    const error = expression('math_arithmetic',{OP:'MINUS'},{
      A:inputBlock(field(expression('api_car_pose',{CAR:varField('car1')}),'x')),
      B:inputBlock(ballCoordinate('x'))});
    const speed = expression('math_arithmetic',{OP:'MULTIPLY'},{A:inputBlock(error),B:inputBlock(variable('response'))});
    const follow = [assign('ball',firstBall()),carBlock('set_velocity','car1',{
      x:num(0),y:inputBlock(expression('math_constrain',{}, {VALUE:inputBlock(speed),LOW:num(-1),HIGH:num(1)})),w:num(0)})];
    const ifBall = expression('controls_if',{}, {
      IF0:inputBlock(compare(ballCount(),'GT',expression('math_number',{NUM:0}))),
      DO0:inputBlock(chain(follow)),ELSE:inputBlock(carBlock('stop'))});
    ifBall.extraState={hasElse:true};
    steps=[connect('car1',1),assign('response',expression('math_number',{NUM:2})),
      expression('api_car_wait_for_position',{CAR:varField('car1')},{timeout:num(10)}),
      forever([assign('balls',expression('api_gps_balls')),ifBall,wait(.05)])];
  } else if(kind==='observe') {
    steps=[expression('controls_whileUntil',{MODE:'WHILE'},{BOOL:{block:expression('logic_boolean',{BOOL:'TRUE'})},DO:inputBlock(chain([
      print(expression('api_gps_feedback_problem')),print(expression('api_gps_cars')),print(expression('api_gps_balls')),wait(.1)
    ]))})];
  } else if(kind==='gamepads' || kind==='keyboard' || kind==='multi') {
    const names=kind==='keyboard'?['car1']:['car1','car2'];
    steps=names.map((name,i)=>connect(name,i+1));
    const loop=[];
    names.forEach((name,pad)=>{
      const axis=(axisName)=>expression('math_arithmetic',{OP:'MULTIPLY'},{A:inputBlock(expression('vsss_pad_axis',{PAD:pad,AXIS:axisName})),B:num(.3)});
      const bipolar=(hi,lo)=>expression('api_input_get_bipolar_ctrl',{}, {high_key:text(hi),low_key:text(lo),keyboard_scale:num(.3),game_scale:num(0)});
      const values=kind==='keyboard'?[bipolar('w','s'),bipolar('a','d'),bipolar('q','e')]:kind==='multi'?[expression('math_number',{NUM:.2}),expression('math_number',{NUM:0}),expression('math_number',{NUM:0})]:[axis('LY'),axis('LX'),axis('RX')];
      loop.push(carBlock('set_velocity',name,{x:inputBlock(values[0]),y:inputBlock(values[1]),w:inputBlock(values[2])}));
    });
    loop.push(wait(kind==='multi'?.1:.05));
    if(kind==='multi') steps.push(expression('controls_repeat_ext',{}, {TIMES:num(10),DO:inputBlock(chain(loop))}));
    else steps.push(expression('controls_whileUntil',{MODE:'WHILE'},{BOOL:{block:expression('logic_boolean',{BOOL:'TRUE'})},DO:inputBlock(chain(loop))}));
  } else if(kind==='velocity') {
    steps=[connect('car1',1),carBlock('set_velocity','car1',{x:num(.2),y:num(0),w:num(0)}),wait(1),carBlock('stop')];
  }
  Blockly.serialization.workspaces.load({variables:ids.map(name=>({id:'v-'+name,name,type:''})),blocks:{languageVersion:0,blocks:[{...chain(steps),x:25,y:25}]}},workspace);
  updateCode();
}
