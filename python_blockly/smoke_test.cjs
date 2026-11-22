// Explicit fake device fixture. These checks never open native input or BLE.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const {spawn,spawnSync} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

(async()=>{
  const server=spawn(process.env.PYTHON||'python',['test_http_server.py','8766'],{cwd:__dirname});
  let browser;
  try{
    await new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error('Server did not start')),10000);
      server.once('error',reject);
      server.once('exit',code=>{if(code)reject(new Error('Server exited '+code));});
      server.stdout.once('data',()=>{clearTimeout(timer);resolve();});
      server.stderr.on('data',data=>process.stderr.write(data));
    });
    browser=await chromium.launch({channel:'msedge',headless:true});
    const page=await browser.newPage({viewport:{width:1500,height:980}});
    const errors=[];
    page.on('pageerror',error=>{errors.push(error.message);process.stderr.write(error.stack+'\n');});
    await page.goto('http://127.0.0.1:8766');
    await page.waitForFunction(()=>typeof workspace!=='undefined'&&workspace&&document.querySelector('#code').textContent.includes('Hello, Python!'));
    assert.equal(await page.locator('#field').count(),1);
    const initial=await page.locator('#code').innerText();
    assert.match(initial,/print\('Hello, Python!'\)/);
    assert.match(initial,/print\(7 \+ 5\)/);
    assert.doesNotMatch(initial,/from vsss|@run|Car\(/);
    assert.equal(await page.locator('#example').count(),0);
    assert.equal(await page.locator('#car-example').count(),0);
    assert.equal(await page.locator('#lesson option[value="blank"]').count(),1);
    for (const lesson of ['kick','goalie']) {
      await page.selectOption('#lesson',lesson);
      const code=await page.locator('#code').innerText();
      assert.doesNotMatch(code,/set_velocity\(x=0\.2/);
      if(lesson==='kick') {
        assert.match(code,/go_to\(x=.*ball.*\.x, y=.*ball.*\.y.*0\.3, angle=90\)/);
        assert.match(code,/set_velocity\(x=1, y=0, w=0\)/);
        assert.ok(code.indexOf('.go_to(')<code.indexOf('.set_velocity('));
      } else {
        assert.match(code,/while True:/);
        assert.match(code,/go_to\(x=.*ball.*\.x, y=-1\.2, angle=90\)/);
        assert.doesNotMatch(code,/set_velocity|response =/);
        assert.match(code,/gps.balls/);
      }
      assert.doesNotMatch(code,/raise |arrival_deadline|fault_name|pose_fresh/);
    }
    assert.equal(await page.evaluate(()=>{
      const before=generatedCode();
      try { loadLesson('missing-example'); return false; }
      catch(error) { return error.message.includes('Unknown example') && before===generatedCode(); }
    }),true);
    const exposedToolbox=await page.evaluate(()=>vsssToolbox);
    const exposed=JSON.stringify(exposedToolbox);
    for(const removed of ['Car settings','set_trim','set_controller','read_trim','read_controller','restart','reconnect_telemetry','gps_config','vsss_invert_y','vsss_data','connection_generation','latest']) assert.ok(!exposed.includes(removed),removed+' is still exposed');
    assert.ok(exposed.includes('Advanced lessons'));
    const inputCategory=exposedToolbox.find(category=>category.name==='Input');
    const advancedCategory=exposedToolbox.find(category=>category.name==='Advanced lessons');
    assert.deepEqual(inputCategory.contents.map(block=>block.type),['vsss_key','vsss_pad_button','vsss_pad_axis']);
    assert.ok(!advancedCategory.contents.some(block=>['vsss_key','vsss_pad_button'].includes(block.type)));
    const keyboardOptions=await page.evaluate(()=>keyboardKeyOptions);
    assert.deepEqual(keyboardOptions[0],['W','w']);
    for(const [label,value] of [['W','w'],['Space','Key.space'],['Enter','Key.enter'],['Arrow up','Key.up']]){
      assert.ok(keyboardOptions.some(option=>option[0]===label&&option[1]===value),label+' keyboard choice missing');
    }
    assert.ok(!keyboardOptions.some(option=>/^F\d+$/.test(option[0])),'function keys should not be offered');
    const inputCode=await page.evaluate(()=>{
      workspace.clear();
      Blockly.serialization.workspaces.load({blocks:{languageVersion:0,blocks:[{type:'vsss_key',fields:{KEY:'Key.left',MODE:'rising_edge'},x:20,y:20}]}},workspace);
      updateCode();
      return generatedCode();
    });
    assert.match(inputCode,/input\.rising_edge\("Key\.left"\)/);
    const runUI=async()=>{
      await page.click('#run');
      await page.waitForFunction(()=>!document.querySelector('#run').disabled&&document.querySelector('#status').textContent.startsWith('Finished'));
    };
    const runCode=code=>page.evaluate(async code=>{
      const started=await request('/run',{code});
      let result;
      do{
        await new Promise(resolve=>setTimeout(resolve,30));
        result=await request('/status',{run_id:started.run_id});
      }while(['running','stopping'].includes(result.state));
      return result;
    },code);
    await page.evaluate(()=>loadLesson('velocity'));
    const downloadPromise=page.waitForEvent('download');
    await page.click('#export');
    const download=await downloadPromise;
    const file=await download.path();
    const exported=fs.readFileSync(file,'utf8');
    const contract=spawnSync(process.env.PYTHON||'python',['verify_export.py',file],{cwd:__dirname,encoding:'utf8'});
    assert.equal(contract.status,0,contract.stdout+contract.stderr);
    process.stdout.write(contract.stdout);
    await runUI();
    assert.match(await page.locator('#output').innerText(),/car1 connected/);
    assert.equal((await runCode(exported)).state,'finished');
    for(const lesson of ['printing','variables','multi']){
      await page.selectOption('#lesson',lesson);
      const code=await page.locator('#code').innerText();
      if(lesson==='multi')assert.match(code,/car2 = Car\(2/);
      const result=await runCode(code);
      assert.equal(result.state,'finished',JSON.stringify(result));
    }
    // Both gamepads and keyboard exports must be valid live programs, stoppable
    // despite their intentionally endless read/control loops.
    for(const lesson of ['rotation','position','waypoints','teleop','read_gamepad','gamepads','keyboard','observe']){
      await page.selectOption('#lesson',lesson);
      const code=await page.locator('#code').innerText();
      if(lesson==='gamepads'){
        assert.match(code,/input.get_axis\("LY0"\)/);
        assert.match(code,/input.get_axis\("LY1"\)/);
        assert.match(code,/car2.set_velocity/);
      }
      if(['observe','position','read_gamepad'].includes(lesson))assert.doesNotMatch(code,/ = Car\(/);
      if(lesson==='read_gamepad'){for(const name of ['A0','B0','X0','Y0','LX0','LY0','RX0','RY0','LT0','RT0'])assert.ok(code.includes(name),name+' missing');assert.doesNotMatch(code,/from vsss import.*Car|gps/);}
      if(lesson==='teleop'){assert.match(code,/high_key=['"]w['"]/);assert.match(code,/low_key=['"]s['"]/);assert.match(code,/keyboard_scale=0.5/);assert.match(code,/input.get_axis\("RX0"\) \* 0.5/);assert.match(code,/y=0/);}
      if(lesson==='rotation'){assert.match(code,/w=0.2/);assert.match(code,/w=-0.2/);assert.match(code,/while True:/);}
      if(lesson==='waypoints'){assert.match(code,/y=0.5/);assert.match(code,/y=-0.5/);assert.match(code,/arrival_status.mode_name == 'reached'/);}
      await page.click('#run');
      await page.waitForFunction(()=>document.querySelector('#status').textContent==='Running');
      await page.waitForTimeout(350);
      if(lesson==='read_gamepad')await page.waitForFunction(()=>document.querySelector('#output').textContent.includes('Triggers LT/RT'));
      assert.equal(await page.locator('#stop').isEnabled(),true);
      await page.click('#stop');
      await page.waitForFunction(()=>document.querySelector('#status').textContent==='Stopped by user.');
    }
    await page.selectOption('#lesson','gamepads');
    assert.equal(await page.evaluate(()=>{
      const code=generatedCode();
      const saved=Blockly.serialization.workspaces.save(workspace);
      workspace.clear();Blockly.serialization.workspaces.load(saved,workspace);updateCode();
      return code===generatedCode();
    }),true);
    // Existing single-car block JSON remains loadable and exports the new API.
    const legacyCode=await page.evaluate(()=>{
      Blockly.serialization.workspaces.load({blocks:{languageVersion:0,blocks:[{
        type:'vsss_car',fields:{NUMBER:1},inputs:{DO:{block:{type:'vsss_velocity',inputs:{
          X:{shadow:{type:'math_number',fields:{NUM:.2}}},Y:{shadow:{type:'math_number',fields:{NUM:0}}},W:{shadow:{type:'math_number',fields:{NUM:0}}}
        },next:{block:{type:'vsss_stop'}}}}}
      }]}},workspace);
      checkCarBlocks(workspace);return generatedCode();
    });
    assert.match(legacyCode,/@run/);
    assert.equal((await runCode(legacyCode)).state,'finished');
    await page.selectOption('#lesson','gamepads');
    // Exercise every generated API block and toolbox flyout: signature discovery
    // must never create malformed definitions or mismatched argument fields.
    const generated=await page.evaluate(()=>{
      const results=[];
      for(const spec of vsssSchema.blocks){
        const block=workspace.newBlock(spec.id);block.initSvg();block.render();
        try{results.push({id:spec.id,code:generatedCode()});}finally{block.dispose();}
      }
      for(const item of workspace.getToolbox().getToolboxItems()){
        if(item.getName&&item.getName()!=='Variables') workspace.getToolbox().setSelectedItem(item);
      }
      workspace.getToolbox().clearSelection();
      return results;
    });
    for(const item of generated){
      const response=await page.evaluate(async code=>{
        // Validate without running any of the advanced configuration blocks.
        return (await fetch('/validate',{method:'POST',headers:{'Content-Type':'application/json','X-VSSS-Blockly':'1'},body:JSON.stringify({code})})).status;
      },item.code);
      assert.equal(response,200,item.id+' did not validate');
    }
    await page.selectOption('#lesson','gamepads');
    await page.screenshot({path:path.join(__dirname,'vsss-preview.png'),fullPage:true});
    await page.selectOption('#lesson','blank');
    assert.equal(await page.evaluate(()=>workspace.getAllBlocks(false).length),0);
    assert.equal(await page.evaluate(()=>workspace.getVariableMap().getAllVariables().length),0);
    assert.equal((await page.locator('#code').innerText()).trim(),'pass');
    await page.selectOption('#lesson','printing');
    await runUI();
    assert.equal(await page.locator('#output').innerText(),'Hello, Python!\n12\n');
    const invalid=await page.evaluate(async()=> (await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json','X-VSSS-Blockly':'1'},body:JSON.stringify({code:'import os'})})).status);
    assert.equal(invalid,400);
    assert.equal(await page.evaluate(async()=> (await fetch('/run',{method:'POST',body:'{}'})).status),403);
    const owner=await browser.newPage();
    await owner.goto('http://127.0.0.1:8766');
    await owner.waitForFunction(()=>typeof workspace!=='undefined'&&workspace);
    await owner.selectOption('#lesson','gamepads');
    await owner.click('#run');
    await owner.waitForFunction(()=>document.querySelector('#output').textContent.includes('car2 connected'));
    await owner.close();
    await page.waitForFunction(async()=> (await request('/status')).state==='stopped',null,{timeout:10000});
    assert.deepEqual(errors,[]);
    console.log('PASS: simplified student permissions/toolbox, Python basics, infinite rotation/waypoints, position reading, multi-car/input examples, export, save/load, Stop and page closure.');
  }finally{
    if(browser)await browser.close();
    server.kill();
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
