const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const Arch = require('../arch.js');
const Diagram = require('../diagram.js');

const FixtureDir = path.join(__dirname, 'fixtures');
const Fixtures = fs.readdirSync(FixtureDir).filter(f => f.endsWith('.json'));

function load(name) {
  return JSON.parse(fs.readFileSync(path.join(FixtureDir, name), 'utf8'));
}

for (const name of Fixtures) {
  test(`figure renders: ${name}`, () => {
    const spec = Arch.normalize(load(name).config);
    const svg = Diagram.render(spec);
    // Same width (so the same on-screen scale) for every model; the two inset columns keep the height close to
    // the block stack's, so switching models barely moves the page.
    const box = svg.match(/^<svg [^>]*viewBox="0 0 (\d+) (\d+)"/);
    assert.strictEqual(box[1], '1740');
    assert.ok(Number(box[2]) <= 1100, `figure height ${box[2]}`);
    assert.ok(svg.endsWith('</svg>'));
    assert.doesNotMatch(svg, /undefined|NaN|null|\[object/);
    assert.ok(svg.includes(`>${spec.layers} x</text>`), 'repetition count missing');
    assert.ok(svg.includes('Token embedding layer'));
    const map = Diagram.layerMap(spec);
    assert.doesNotMatch(map, /undefined|NaN/);
    assert.strictEqual((map.match(/<rect /g) || []).length % spec.layers, 0);
  });
}

test('figure labels follow the config', () => {
  const find = key => Fixtures.find(f => f.includes(key));
  // Mixed layer kinds get one colored box each, with their layer counts.
  const gemma3Spec = Arch.normalize(load(find('gemma-3')).config);
  const gemma3 = Diagram.render(gemma3Spec);
  const slidingCount = gemma3Spec.layerTypes.filter(t => t.mixer === 'sliding').length;
  assert.ok(gemma3.includes('class="b-k-sliding"') && gemma3.includes('class="b-k-full"'));
  assert.ok(gemma3.includes(`>${slidingCount} x</text>`));
  assert.ok(gemma3.includes('Post-RMSNorm 1'));
  assert.ok(gemma3.includes('QK-Norm'));

  const deepseek = Diagram.render(Arch.normalize(load(find('deepseek-v3')).config));
  assert.ok(deepseek.includes('MoE layer'));
  assert.ok(deepseek.includes('shared expert'));
  assert.match(deepseek, /First <tspan class="hl">3<\/tspan> blocks use/);
  assert.ok(deepseek.includes('18,432'), 'layer widths must stay exact, not 18K');

  const gpt2 = Diagram.render(Arch.normalize(load(find('gpt2')).config));
  assert.ok(gpt2.includes('Positional'));
  assert.ok(gpt2.includes('FeedForward (GELU) module'));
  assert.ok(!gpt2.includes('RoPE'));

  const next = Diagram.render(Arch.normalize(load(find('qwen3-next')).config));
  assert.ok(next.includes('class="b-k-linear"') && next.includes('DeltaNet'));
  assert.ok(next.includes('>36 x</text>') && next.includes('>12 x</text>'));

  // A dense model and an MoE model must differ in shape, not only in labels.
  const dense = Diagram.render(Arch.normalize(load(find('qwen3-8b')).config));
  const sparse = Diagram.render(Arch.normalize(load(find('qwen3-30b')).config));
  const rects = svg => (svg.match(/<rect /g) || []).length;
  assert.ok(rects(sparse) > rects(dense) + 100, 'MoE figure should draw every expert');
});

test('garbage config still draws', () => {
  const svg = Diagram.render(Arch.normalize({ model_type: 'weird' }));
  assert.match(svg, /^<svg /);
});
