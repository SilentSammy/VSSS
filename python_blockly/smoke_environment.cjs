// Browser + real local simulator/RPC. Never selects Live or imports input devices.
const {chromium}=require('playwright');
const {spawn,spawnSync}=require('node:child_process');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');

(async()=>{
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'vsss-blockly-browser-'));
  const endpoint=path.join(directory,'rpc.json');
  const python=process.env.PYTHON||'python';
  const server=spawn(python,['server.py','--no-browser','--port','8767','--provider-port','0','--endpoint',endpoint],{cwd:__dirname});
  let browser;
  const errors=[];
  server.stderr.on('data',()=>{});
  function car() {
    const result=spawnSync(python,['-c',
      'import json,sys; from vsss import VSSSProvider; from vsss.plot import make_plotter; p=make_plotter(width=480); c=VSSSProvider.connect(sys.argv[1]); s=c.gps.state(); q=s.cars[0].pose; print(json.dumps(dict(x=q.x,y=q.y,px=p.width/2+q.x*p.scale,py=p.height/2-q.y*p.scale,scale=p.scale,source=c.describe()["source_id"]))); c.close()',endpoint],{cwd:path.dirname(__dirname),encoding:'utf8'});
    assert.equal(result.status,0,result.stderr);
    return JSON.parse(result.stdout);
  }
  try {
    await new Promise((resolve,reject)=>{server.stdout.once('data',resolve);server.once('error',reject);});
    browser=await chromium.launch({channel:'msedge',headless:true});
    const page=await browser.newPage({viewport:{width:1600,height:1100}});
    page.on('pageerror',e=>errors.push(e.message));
    // Initial connection failure must recover without reloading the page.
    let firstEnvironmentRequest=true;
    await page.route('**/api/environment', route=>{
      if (firstEnvironmentRequest) { firstEnvironmentRequest=false; return route.abort('connectionfailed'); }
      return route.continue();
    });
    await page.goto('http://127.0.0.1:8767');
    await page.waitForFunction(()=>typeof workspace!=='undefined'&&workspace&&document.querySelector('#environment').options.length>=4);
    assert.equal(await page.locator('#environment').inputValue(),'car1_ball1');
    assert.deepEqual(await page.locator('#environment option').allTextContents(),[
      'Live','1 car','1 car 1 ball','2 cars 1 ball','4 cars 1 ball','goalie easy','goalie hard']);
    assert.equal(fs.existsSync(endpoint),false,'Page load must not start hardware or a provider');
    await page.route('**/never-resolves', ()=>{});
    const timedOut=await page.evaluate(async()=>{
      try {await apiJSON('/never-resolves',{},50); return false;}
      catch(error){return error.message.includes('timed out');}
    });
    assert.equal(timedOut,true,'Hung requests must have a bounded timeout');
    await page.unroute('**/never-resolves');
    await page.route('**/environment', route=>route.abort('connectionfailed'));
    await page.click('#reset-environment');
    await page.waitForFunction(()=>!document.querySelector('#environment').disabled);
    assert.match(await page.locator('#environment-status').innerText(),/local Blockly server/);
    await page.unroute('**/environment');
    await page.click('#reset-environment');
    await page.waitForFunction(()=>document.querySelector('#field').classList.contains('editable'));
    await page.click('#field-collapse');
    assert.equal(await page.locator('#field').isVisible(),false);
    assert.equal(await page.locator('#field-collapse').innerText(),'Expand');
    await page.click('#field-collapse');
    assert.equal(await page.locator('#field').isVisible(),true);
    await page.click('#field-fullscreen');
    await page.waitForFunction(()=>document.fullscreenElement?.id==='field-panel');
    assert.equal(await page.locator('#field-fullscreen').innerText(),'Exit full screen');
    await page.click('#field-fullscreen');
    await page.waitForFunction(()=>!document.fullscreenElement);
    const before=car();
    assert.ok(Math.abs(before.y)<.01);
    const canvas=page.locator('#field');
    await canvas.scrollIntoViewIfNeeded();
    const box=await canvas.boundingBox();
    const dimensions=await canvas.evaluate(el=>({width:el.width,height:el.height}));
    const x=box.x+before.px*box.width/dimensions.width;
    const y=box.y+before.py*box.height/dimensions.height;
    await page.mouse.move(x,y);await page.mouse.down();
    await page.mouse.move(x+.2*before.scale*box.width/dimensions.width,y,{steps:6});
    await page.waitForTimeout(300);await page.mouse.up();
    await page.waitForTimeout(300);
    assert.ok(Math.abs(car().x-(before.x+.2))<.03,'Canvas drag must update the real simulator pose');
    await page.route('**/run', route=>route.abort('connectionfailed'));
    await page.click('#run');
    await page.waitForFunction(()=>document.querySelector('#status').textContent==='Could not start');
    assert.equal(await page.locator('#environment').isEnabled(),true);
    await page.unroute('**/run');
    await page.selectOption('#environment','car1_ball1');
    await page.waitForFunction(()=>!document.querySelector('#environment').disabled);
    const changed=car();assert.equal(changed.source,before.source);
    assert.ok(Math.abs(changed.x+.2)<.01,'Selection must reset the existing simulator');
    await page.click('#run');
    await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Finished'));
    assert.equal(await page.locator('#output').innerText(),'Hello, Python!\n12\n');
    assert.ok(Math.abs(car().x+.2)<.01,'Run must not reset the scenario');
    // Execute the actual blocks against the simulator: line up, strike +Y, stop.
    await page.selectOption('#lesson','kick');
    await page.click('#run');
    await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Finished'),null,{timeout:20000});
    assert.ok(!((await page.locator('#output').innerText()).includes('Error')));
    assert.match(await page.locator('#output').innerText(),/Aligning 30 cm behind the ball[\s\S]*Aligned\. Kicking at full speed!/);
    const kicked=spawnSync(python,['-c',
      'import sys; from vsss import VSSSProvider; c=VSSSProvider.connect(sys.argv[1]); print(c.gps.state().balls[0].pose.y); c.close()',endpoint],{cwd:path.dirname(__dirname),encoding:'utf8'});
    assert.equal(kicked.status,0,kicked.stderr);
    assert.ok(Number(kicked.stdout)>.4,'Kick example must move the ball toward the top wall');
    await page.selectOption('#environment','goalie_easy');
    await page.waitForFunction(()=>!document.querySelector('#environment').disabled);
    const goalieStart=car();
    await page.selectOption('#lesson','goalie');
    await page.click('#run');
    await page.waitForFunction(()=>document.querySelector('#status').textContent==='Running');
    await page.waitForTimeout(1200);
    assert.ok(Math.abs(car().x-goalieStart.x)>.01,'Goalie must move sideways toward the incoming ball');
    await page.click('#stop');
    await page.waitForFunction(()=>document.querySelector('#status').textContent==='Stopped by user.');
    // Lose the heartbeat mid-run: controls must recover once its lease ends.
    await page.click('#run');
    await page.waitForFunction(()=>document.querySelector('#status').textContent==='Running');
    await page.route('**/status', route=>route.abort('connectionfailed'));
    await page.waitForFunction(()=>!document.querySelector('#environment').disabled,null,{timeout:15000});
    await page.unroute('**/status');
    await page.selectOption('#environment','cars4_ball1');
    await page.waitForFunction(()=>!document.querySelector('#environment').disabled);
    assert.equal(await page.locator('#environment').inputValue(),'cars4_ball1');
    await page.selectOption('#lesson','rotation');
    await page.click('#run');
    await page.waitForFunction(()=>document.querySelector('#status').textContent==='Running');
    assert.equal(await page.locator('#environment').isDisabled(),true);
    assert.equal(await page.evaluate(async()=>{
      try {await request('/environment',{selection:'goalie_easy'});return false;}
      catch(error){return error.message.includes('Stop the program');}
    }),true);
    await page.waitForTimeout(300);
    await page.click('#stop');
    await page.waitForFunction(()=>document.querySelector('#status').textContent==='Stopped by user.');
    await page.selectOption('#lesson','printing');
    await page.screenshot({path:path.join(__dirname,'environment-preview.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('PASS: environment menu, simulator launch/reset/reuse, real RPC canvas drag, student Run, and embedded plot.');
  } finally {
    if(browser)await browser.close();
    server.kill();
    await new Promise(resolve=>{if(server.exitCode!==null)resolve();else server.once('exit',resolve);});
    fs.rmSync(directory,{recursive:true,force:true});
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
