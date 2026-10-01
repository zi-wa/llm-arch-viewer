const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const Arch = require('../arch.js');
const Inset = require('../inset_mixer.js');

const FixtureDir = path.join(__dirname, 'fixtures');
const X0 = 20, Y0 = 50, W = 455;

function load(name) {
  return Arch.normalize(JSON.parse(fs.readFileSync(path.join(FixtureDir, name), 'utf8')).config);
}

function mixerFixtures() {
  return fs.readdirSync(FixtureDir)
    .map(f => ({ file: f, spec: load(f) }))
    .filter(({ spec }) => spec.linearMixer);
}

function xAttrs(svg) {
  const xs = [];
  for (const m of svg.matchAll(/[ "](?:x|cx|x1|x2)="(-?[\d.]+)"/g)) xs.push(Number(m[1]));
  for (const m of svg.matchAll(/<path d="M([^"]+)"/g)) {
    const nums = m[1].split(/[ML, ]+/).filter(Boolean).map(Number);
    for (let i = 0; i < nums.length; i += 2) xs.push(nums[i]);
  }
  return xs;
}

for (const { file, spec } of mixerFixtures()) {
  const kinds = [...new Set(spec.layerTypes.map(t => t.mixer))].filter(k => ['linear', 'mamba', 'conv'].includes(k));
  for (const kind of kinds) {
    test(`mixer inset renders: ${file} (${kind})`, () => {
      const out = Inset.draw(spec, { x0: X0, y0: Y0, w: W, kind });
      const svg = out.svg.join('');
      assert.doesNotMatch(svg, /undefined|NaN/);
      assert.ok(out.y1 > Y0);
      assert.ok(out.anchor && out.anchor.x === X0);
      for (const x of xAttrs(svg)) {
        assert.ok(x >= X0 - 0.5 && x <= X0 + W + 0.5, `x ${x} within frame in ${file}`);
      }
    });
  }
}

test('Gated DeltaNet module labels heads and head dim', () => {
  const spec = load(fs.readdirSync(FixtureDir).find(f => f.includes('qwen3-next')));
  const out = Inset.draw(spec, { x0: X0, y0: Y0, w: W, kind: 'linear' });
  const svg = out.svg.join('');
  assert.ok(svg.includes('Gated DeltaNet module'));
  assert.ok(svg.includes('32') && svg.includes('128'));
});

test('Mamba and Mamba-2 titles differ', () => {
  const mamba = load(fs.readdirSync(FixtureDir).find(f => f.includes('mamba-130m')));
  const mamba2 = load(fs.readdirSync(FixtureDir).find(f => f.includes('granite-4-0-h-tiny')));
  const outMamba = Inset.draw(mamba, { x0: X0, y0: Y0, w: W, kind: 'mamba' }).svg.join('');
  const outMamba2 = Inset.draw(mamba2, { x0: X0, y0: Y0, w: W, kind: 'mamba' }).svg.join('');
  assert.ok(outMamba.includes('Mamba module') && !outMamba.includes('Mamba-2 module'));
  assert.ok(outMamba2.includes('Mamba-2 module'));
  assert.ok(outMamba2.includes('48') && outMamba2.includes('64'), 'Mamba-2 shows heads x head dim');
});

test('short convolution module names its kernel', () => {
  const spec = load(fs.readdirSync(FixtureDir).find(f => f.includes('lfm2')));
  const svg = Inset.draw(spec, { x0: X0, y0: Y0, w: W, kind: 'conv' }).svg.join('');
  assert.ok(svg.includes('Short convolution module'));
  assert.ok(svg.includes('class="b-k-conv"'));
  assert.ok(svg.includes('3'), 'conv kernel size shown');
});

test('unknown mixer kind falls back to a generic block without throwing', () => {
  const spec = load(fs.readdirSync(FixtureDir).find(f => f.includes('lfm2')));
  const out = Inset.draw(spec, { x0: X0, y0: Y0, w: W, kind: 'quantum' });
  const svg = out.svg.join('');
  assert.doesNotMatch(svg, /undefined|NaN/);
  assert.ok(svg.includes('module'));
});
