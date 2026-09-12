const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const handler = source.slice(source.indexOf(" $('agent-plan').onclick="), source.indexOf(' function applyAgentConfig()'));

function page(post) {
  const elements = new Map();
  const timers = new Set();
  const traces = [];
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: id === 'agent-mode' ? 'codex' : '20 W 加热', disabled: false, hidden: false, textContent: '',
      classList: {add() {}, toggle() {}},
    });
    return elements.get(id);
  };
  const original = {model_id: 'model-1', duration_s: 60};
  const context = vm.createContext({
    $: element, cfg: structuredClone(original), model: {id: 'model-1', name: 'Test model'},
    agentConfig: {model_id: 'model-1', duration_s: 300},
    clone: structuredClone, post, protect: fn => fn, agentModeLabels: {codex: 'Codex'},
    agentTraceReset: () => {}, agentTraceSet: (...args) => traces.push(args),
    setInterval: fn => {timers.add(fn); return fn;}, clearInterval: fn => timers.delete(fn),
  });
  vm.runInContext(handler, context);
  return {context, element, timers, traces, original};
}

test('timeout keeps current configuration and disables any previous draft', async () => {
  let complete;
  const p = page(() => new Promise(resolve => {complete = resolve;}));
  const request = p.element('agent-plan').onclick();
  assert.equal(p.context.agentConfig, null);
  assert.equal(p.element('agent-confirm').disabled, true);
  complete({ok: false, error_code: 'timeout', questions: ['Codex 上游连接超时']});
  await request;
  assert.deepEqual(p.context.cfg, p.original);
  assert.equal(p.context.agentConfig, null);
  assert.equal(p.element('agent-apply').disabled, true);
  assert.equal(p.element('agent-confirm').disabled, true);
  assert.match(p.element('agent-plan-text').textContent, /配置生成未完成/);
  assert.match(p.element('agent-plan-questions').textContent, /上游连接超时/);
  assert.equal(p.traces.some(([,state]) => state === 'done'), false);
  assert.equal(p.timers.size, 0);
});

test('successful generation presents a draft without applying or simulating it', async () => {
  const draft = {model_id: 'model-1', duration_s: 600, heat_sources: [{power_W:20, faces:[1]}]};
  const p = page(async () => ({ok: true, config: draft, changes: ['时长 600 秒']}));
  await p.element('agent-plan').onclick();
  assert.deepEqual(p.context.cfg, p.original);
  assert.deepEqual(p.context.agentConfig, draft);
  assert.equal(p.element('agent-confirm').disabled, false);
  assert.equal(p.element('agent-apply').disabled, false);
  assert.equal(p.element('agent-plan').disabled, false);
  assert.equal(p.timers.size, 0);
});

test('HTTP failure clears stale drafts and stops waiting feedback', async () => {
  const p = page(async () => {throw new Error('Failed to fetch');});
  await assert.rejects(p.element('agent-plan').onclick(), /Failed to fetch/);
  assert.equal(p.context.agentConfig, null);
  assert.equal(p.element('agent-confirm').disabled, true);
  assert.equal(p.element('agent-plan').disabled, false);
  assert.equal(p.timers.size, 0);
  assert.equal(p.element('agent-plan-questions').textContent, 'Failed to fetch');
});

test('an ok response without a heat source cannot enable confirmation', async () => {
  const p = page(async () => ({ok: true, config: {model_id:'model-1', heat_sources:[]}}));
  await assert.rejects(p.element('agent-plan').onclick(), /没有生成热源/);
  assert.deepEqual(p.context.cfg, p.original);
  assert.equal(p.context.agentConfig, null);
  assert.equal(p.element('agent-confirm').disabled, true);
  assert.equal(p.element('agent-apply').disabled, true);
});
