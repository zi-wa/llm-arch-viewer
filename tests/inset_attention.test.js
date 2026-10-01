const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const Arch = require('../arch.js');
const AttentionInset = require('../inset_attention.js');

const X0 = 20, Y0 = 50, W = 455;

// Kit.text/rich() turns a '{value}' marker into this tspan; assertions match the rendered form, not the source marker.
const hl = (v) => `<tspan class="hl">${Number(v).toLocaleString('en-US')}</tspan>`;

function fixtureSpec(file) {
  const raw = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', file), 'utf8'));
  return Arch.normalize(raw.config);
}

function attnKindsOf(spec) {
  return [...new Set(spec.layerTypes.map((lt) => lt.mixer))].filter((m) => ['full', 'sliding', 'chunked'].includes(m));
}

function assertWithinBounds(svg) {
  const xs = svg.join('').match(/x="(-?[\d.]+)"/g) || [];
  for (const attr of xs) {
    const value = Number(attr.slice(3, -1));
    assert.ok(value >= X0 - 0.01 && value <= X0 + W + 0.01, `x=${value} outside [${X0}, ${X0 + W}]`);
  }
}

function assertClean(svg) {
  const joined = svg.join('');
  assert.ok(!joined.includes('undefined'), 'svg contains undefined');
  assert.ok(!joined.includes('NaN'), 'svg contains NaN');
}

test('every fixture with attention renders without throwing', () => {
  for (const file of fs.readdirSync(path.join(__dirname, 'fixtures'))) {
    const spec = fixtureSpec(file);
    const kinds = attnKindsOf(spec);
    if (!kinds.length && spec.attention.kind !== 'MLA') continue;
    const out = AttentionInset.draw(spec, { x0: X0, y0: Y0, w: W, kinds: [...new Set(spec.layerTypes.map((lt) => lt.mixer))] });
    assert.ok(Array.isArray(out.svg) && out.svg.length > 0, `${file}: no svg produced`);
    assert.ok(out.y1 > Y0, `${file}: y1 did not advance`);
    assert.ok(out.anchor && typeof out.anchor.x === 'number' && typeof out.anchor.y === 'number', `${file}: bad anchor`);
    assertClean(out.svg);
    assertWithinBounds(out.svg);
  }
});

test('MHA fixtures (gpt2-xl, olmo-2) look 1:1', () => {
  for (const file of ['openai-community-gpt2-xl.json', 'allenai-olmo-2-1124-7b.json']) {
    const spec = fixtureSpec(file);
    assert.strictEqual(spec.attention.kind, 'MHA');
    const out = AttentionInset.draw(spec, { x0: X0, y0: Y0, w: W, kinds: ['full'] });
    const joined = out.svg.join('');
    assert.ok(joined.includes(hl(spec.attention.heads)), `${file}: missing exact head count in label`);
  }
});

test('GQA fixtures (qwen3-8b, llama-3.2-1b) label KV heads and group size', () => {
  for (const file of ['qwen-qwen3-8b.json', 'unsloth-llama-3-2-1b.json']) {
    const spec = fixtureSpec(file);
    assert.strictEqual(spec.attention.kind, 'GQA');
    const groupSize = spec.attention.heads / spec.attention.kvHeads;
    const out = AttentionInset.draw(spec, { x0: X0, y0: Y0, w: W, kinds: ['full'] });
    const joined = out.svg.join('');
    assert.ok(joined.includes(hl(spec.attention.kvHeads)), `${file}: missing KV head count`);
    assert.ok(joined.includes(hl(groupSize)), `${file}: missing group size`);
  }
});

test('synthetic MQA renders an all-to-one shape with exact counts', () => {
  const base = fixtureSpec('unsloth-llama-3-2-1b.json');
  const spec = {
    ...base,
    attention: { ...base.attention, kind: 'MQA', heads: 32, kvHeads: 1, headDim: 128, mla: null },
    layerTypes: base.layerTypes.map((lt) => ({ ...lt, mixer: 'full' })),
  };
  const out = AttentionInset.draw(spec, { x0: X0, y0: Y0, w: W, kinds: ['full'] });
  const joined = out.svg.join('');
  assert.ok(joined.includes('Multi-query attention'));
  assert.ok(joined.includes(`${hl(32)} query heads`));
  assert.ok(joined.includes(`${hl(1)} KV head`));
  assertClean(out.svg);
  assertWithinBounds(out.svg);
});

test('synthetic MQA with a very large head count falls back to truncation without breaking bounds', () => {
  const base = fixtureSpec('unsloth-llama-3-2-1b.json');
  const spec = {
    ...base,
    attention: { ...base.attention, kind: 'MQA', heads: 256, kvHeads: 1, headDim: 64, mla: null },
    layerTypes: base.layerTypes.map((lt) => ({ ...lt, mixer: 'full' })),
  };
  const out = AttentionInset.draw(spec, { x0: X0, y0: Y0, w: W, kinds: ['full'] });
  const joined = out.svg.join('');
  assert.ok(joined.includes(`${hl(256)} query heads`));
  assert.ok(joined.includes('...'), 'expected an ellipsis for the truncated head row');
  assertClean(out.svg);
  assertWithinBounds(out.svg);
});

test('MLA fixtures (deepseek-v3, kimi-k2) state the computed cache values', () => {
  for (const file of ['deepseek-ai-deepseek-v3.json', 'moonshotai-kimi-k2-instruct.json']) {
    const spec = fixtureSpec(file);
    assert.strictEqual(spec.attention.kind, 'MLA');
    const mla = spec.attention.mla;
    const mlaCache = mla.kvLoraRank + mla.qkRopeHeadDim;
    const mhaEquivalent = spec.attention.heads * (mla.qkNopeHeadDim + mla.qkRopeHeadDim + mla.vHeadDim);
    const out = AttentionInset.draw(spec, { x0: X0, y0: Y0, w: W, kinds: ['full'] });
    const joined = out.svg.join('');
    assert.ok(joined.includes('Multi-head latent attention'));
    assert.ok(joined.includes(hl(mlaCache)), `${file}: missing MLA cache value`);
    assert.ok(joined.includes(mhaEquivalent.toLocaleString('en-US')), `${file}: missing MHA-equivalent cache value`);
    assert.ok(joined.includes(hl(mla.kvLoraRank)), `${file}: missing latent rank`);
    assertClean(out.svg);
    assertWithinBounds(out.svg);
  }
});

test('sliding-window fixtures (mistral-7b, gemma-2, gemma-3, gpt-oss) draw exactly one local mask with the real window', () => {
  for (const file of ['mistralai-mistral-7b-v0-1.json', 'unsloth-gemma-2-9b.json', 'unsloth-gemma-3-4b-it.json', 'openai-gpt-oss-20b.json']) {
    const spec = fixtureSpec(file);
    const kinds = attnKindsOf(spec);
    const out = AttentionInset.draw(spec, { x0: X0, y0: Y0, w: W, kinds });
    const joined = out.svg.join('');
    const windowLabel = `window ${hl(spec.attention.slidingWindow)}`;
    const occurrences = joined.split('>local</tspan>').length - 1;
    assert.strictEqual(occurrences, 1, `${file}: expected exactly one local mask`);
    assert.ok(joined.includes(windowLabel), `${file}: missing real window value`);
  }
});

test('llama-4 (chunked) draws exactly one chunked mask with the real chunk size', () => {
  const spec = fixtureSpec('unsloth-llama-4-scout-17b-16e-instruct.json');
  const kinds = attnKindsOf(spec);
  const out = AttentionInset.draw(spec, { x0: X0, y0: Y0, w: W, kinds });
  const joined = out.svg.join('');
  const occurrences = joined.split('>chunked</tspan>').length - 1;
  assert.strictEqual(occurrences, 1, 'expected exactly one chunked mask');
  assert.ok(joined.includes(`chunk ${hl(spec.attention.chunkSize)}`));
});

test('gpt-oss sinks add a highlighted sink column to every drawn mask', () => {
  const spec = fixtureSpec('openai-gpt-oss-20b.json');
  assert.strictEqual(spec.attention.sinks, true);
  const kinds = attnKindsOf(spec);
  const out = AttentionInset.draw(spec, { x0: X0, y0: Y0, w: W, kinds });
  const joined = out.svg.join('');
  const sinkFill = 'fill="#5646d6"';
  assert.ok(joined.includes(sinkFill), 'expected an accent-colored sink column');
});

test('mixed local/global layers label a ratio', () => {
  const spec = fixtureSpec('unsloth-gemma-3-4b-it.json');
  const kinds = attnKindsOf(spec);
  const out = AttentionInset.draw(spec, { x0: X0, y0: Y0, w: W, kinds });
  const joined = out.svg.join('');
  assert.ok(joined.includes(`per ${hl(6)} layers`), 'expected the 5:1-per-6 pattern gemma-3 actually uses');
  assert.ok(joined.includes(`${hl(5)} local`));
  assert.ok(joined.includes(`${hl(1)} global`));
});

test('a single attention kind draws no ratio label', () => {
  const spec = fixtureSpec('qwen-qwen3-8b.json');
  const out = AttentionInset.draw(spec, { x0: X0, y0: Y0, w: W, kinds: ['full'] });
  const joined = out.svg.join('');
  assert.ok(!joined.includes('per <tspan'), 'a single-kind model should not print a mixing ratio');
});

test('non-attention-only mixer kinds do not add a mask panel', () => {
  const spec = fixtureSpec('state-spaces-mamba-130m-hf.json');
  assert.strictEqual(attnKindsOf(spec).length, 0);
});

test('hybrid attention+SSM fixtures (granite, lfm2, qwen3-next) still draw a clean head panel', () => {
  for (const file of ['ibm-granite-granite-4-0-h-tiny.json', 'liquidai-lfm2-1-2b.json', 'qwen-qwen3-next-80b-a3b-instruct.json']) {
    const spec = fixtureSpec(file);
    const kinds = [...new Set(spec.layerTypes.map((lt) => lt.mixer))];
    const out = AttentionInset.draw(spec, { x0: X0, y0: Y0, w: W, kinds });
    assertClean(out.svg);
    assertWithinBounds(out.svg);
  }
});
