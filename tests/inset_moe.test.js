const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const Arch = require('../arch.js');
const Inset = require('../inset_moe.js');

const FixtureDir = path.join(__dirname, 'fixtures');
const X0 = 20, Y0 = 50, W = 455;

function load(name) {
  return Arch.normalize(JSON.parse(fs.readFileSync(path.join(FixtureDir, name), 'utf8')).config);
}

function moeFixtures() {
  return fs.readdirSync(FixtureDir)
    .map(f => ({ file: f, spec: load(f) }))
    .filter(({ spec }) => spec.ffn.moe);
}

for (const { file, spec } of moeFixtures()) {
  test(`moe inset renders: ${file}`, () => {
    const out = Inset.draw(spec, { x0: X0, y0: Y0, w: W });
    const svg = out.svg.join('');
    assert.doesNotMatch(svg, /undefined|NaN/);

    const rectFills = [...svg.matchAll(/<rect x="([-\d.]+)"[^>]*fill="(#[0-9a-f]+)"/g)];
    const cellFills = rectFills.filter(m => m[2] === '#e9e9e9' || m[2] === '#5646d6');
    assert.strictEqual(cellFills.length, spec.ffn.moe.experts, 'one cell per routed expert');
    const activeCells = cellFills.filter(m => m[2] === '#5646d6');
    assert.strictEqual(activeCells.length, spec.ffn.moe.topK, 'topK cells marked active');

    for (const m of rectFills) {
      const x = Number(m[1]);
      assert.ok(x >= X0 - 0.5 && x <= X0 + W + 0.5, `x ${x} within frame`);
    }
    assert.ok(out.y1 > Y0);
    assert.ok(out.anchor && out.anchor.x === X0);
    assert.ok(out.expertTop && Number.isFinite(out.expertTop.x) && Number.isFinite(out.expertTop.y));
  });
}

test('expert counts differ visibly in cell size across scale', () => {
  const find = key => moeFixtures().find(({ file }) => file.includes(key));
  const small = Inset.draw(find('mixtral').spec, { x0: X0, y0: Y0, w: W });
  const big = Inset.draw(find('qwen3-next').spec, { x0: X0, y0: Y0, w: W });
  const cellSize = out => {
    const widths = [...out.svg.join('').matchAll(/<rect x="[-\d.]+" y="[-\d.]+" width="([\d.]+)" height="([\d.]+)"/g)]
      .filter(m => m[1] === m[2] && Number(m[1]) <= 34)
      .map(m => Number(m[1]));
    return Math.min(...widths);
  };
  assert.ok(cellSize(small) > cellSize(big), 'more experts should draw smaller cells');
});

test('shared experts render as separate dark cells', () => {
  const deepseek = load(fs.readdirSync(FixtureDir).find(f => f.includes('deepseek-v3')));
  const out = Inset.draw(deepseek, { x0: X0, y0: Y0, w: W });
  const svg = out.svg.join('');
  assert.ok(svg.includes('#2b2b2b'), 'shared cell uses Palette.dark');
  assert.match(svg, />shared</);
});
