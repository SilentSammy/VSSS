// All browser runs use an explicit fake-hardware fixture, never server.py.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const {spawn, spawnSync} = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');

(async () => {
  const server = spawn(process.env.PYTHON || 'python', ['test_http_server.py', '8766'], {cwd: __dirname});
  let browser;
  try {
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Server did not start')), 10000);
      server.once('error', reject);
      server.stdout.on('data', () => {clearTimeout(timeout); resolve();});
      server.stderr.on('data', data => process.stderr.write(data));
    });
    browser = await chromium.launch({channel: 'msedge', headless: true});
    const page = await browser.newPage({viewport: {width: 1400, height: 900}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8766');
    await page.waitForFunction(() => document.querySelector('#code').textContent.includes('with Car(1)'));
    assert.equal(await page.locator('#field').count(), 0);
    assert.match(await page.locator('#run').innerText(), /Run live/);
    await page.click('#run');
    await page.waitForFunction(() => document.querySelector('#output').textContent.includes('Connected.'));
    assert.equal(await page.locator('#run').isDisabled(), true);
    assert.equal(await page.locator('#stop').isEnabled(), true);
    await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('Finished'));
    assert.match(await page.locator('#output').innerText(), /Pose\(x=0.1, y=0.2/);
    assert.match(await page.locator('#code').innerText(), /car\.set_velocity\(x=0.2, y=0, w=0\)/);
    const downloaded = page.waitForEvent('download');
    await page.click('#export');
    const download = await downloaded;
    const exported = fs.readFileSync(await download.path(), 'utf8');
    assert.match(exported, /from car import Car/);
    assert.match(exported, /from time import sleep/);
    assert.doesNotMatch(exported, /simulator/);
    const contract = spawnSync(process.env.PYTHON || 'python', ['verify_export.py', await download.path()], {cwd: __dirname, encoding: 'utf8'});
    assert.equal(contract.status, 0, contract.stdout + contract.stderr);
    process.stdout.write(contract.stdout);
    // Stop must interrupt a long program through the actual browser control.
    await page.evaluate(() => {
      workspace.getAllBlocks(false).find(b => b.type === 'math_number' && b.getFieldValue('NUM') === 1).setFieldValue(60, 'NUM');
      updateCode();
    });
    await page.click('#run');
    await page.waitForFunction(() => document.querySelector('#output').textContent.includes('Connected.'));
    await page.click('#stop');
    await page.waitForFunction(() => document.querySelector('#status').textContent === 'Stopped by user.');
    assert.equal(await page.locator('#run').isEnabled(), true);
    await page.screenshot({path: path.join(__dirname, 'vsss-preview.png'), fullPage: true});
    // Car blocks outside a session fail before any hardware command.
    await page.evaluate(() => {
      workspace.clear(); workspace.newBlock('vsss_stop').initSvg(); updateCode();
    });
    await page.click('#run');
    assert.match(await page.locator('#output').innerText(), /inside the/);
    await page.click('#example');
    await page.waitForFunction(() => document.querySelector('#code').textContent.includes('range(3)'));
    await page.click('#run');
    await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('Finished'));
    assert.equal(await page.locator('#output').innerText(), 'Hello from Python!\nHello from Python!\nHello from Python!\n12\n');
    await page.evaluate(() => {
      workspace.getAllBlocks(false).find(b => b.type === 'math_number' && b.getFieldValue('NUM') === 3).setFieldValue(2, 'NUM');
      updateCode();
    });
    await page.click('#run');
    await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('Finished'));
    assert.equal(await page.locator('#output').innerText(), 'Hello from Python!\nHello from Python!\n12\n');
    assert.equal(await page.evaluate(() => {
      const code = generatedCode();
      const saved = Blockly.serialization.workspaces.save(workspace);
      workspace.clear();
      Blockly.serialization.workspaces.load(saved, workspace);
      return generatedCode() === code;
    }), true);
    // API validation and worker runtime errors still reach the live UI.
    const failed = await page.evaluate(async () => {
      const started = await request('/run', {code: 'print(1 / 0)'});
      let result;
      do {
        await new Promise(resolve => setTimeout(resolve, 50));
        result = await request('/status', {run_id: started.run_id});
      } while (['running', 'stopping'].includes(result.state));
      return result;
    });
    assert.equal(failed.state, 'failed');
    assert.match(failed.error, /ZeroDivisionError/);
    const rejected = await page.evaluate(async () => {
      const response = await fetch('/run', {method: 'POST', headers: {
        'Content-Type': 'application/json', 'X-Blockly-Demo': '1'
      }, body: JSON.stringify({code: 'import os'})});
      return response.status;
    });
    assert.equal(rejected, 400);
    assert.equal(await page.evaluate(async () => (await fetch('/run', {method: 'POST', body: '{}'})).status), 403);
    // Closing the owning page must end a live run, with lease expiry as fallback.
    const owner = await browser.newPage();
    await owner.goto('http://127.0.0.1:8766');
    await owner.waitForFunction(() => document.querySelector('#code').textContent.includes('with Car(1)'));
    await owner.evaluate(() => {
      workspace.getAllBlocks(false).find(b => b.type === 'math_number' && b.getFieldValue('NUM') === 1).setFieldValue(60, 'NUM');
      updateCode();
    });
    await owner.click('#run');
    await owner.waitForFunction(() => document.querySelector('#output').textContent.includes('Connected.'));
    await owner.close();
    await page.waitForFunction(async () => {
      const result = await request('/status'); // Observer does not renew the lease.
      return result.state === 'stopped';
    }, null, {timeout: 10000});
    assert.deepEqual(errors, []);
    console.log('PASS: live UI, real-time execution protocol with fake I/O, Stop, page close, Python export, car scope, block edits, save/load, errors, request guards; no simulator view.');
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
