// Renders one inset module for every fixture into logs/preview_<module>.html and screenshots it with headless
// Chrome into logs/preview_<module>.png, for eyeballing layouts while building an inset.
// Run: node tests/preview_insets.js attention|moe|mixer [fixture-substring]
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const Arch = require('../arch.js');
const Kit = require('../svgkit.js');

const Root = path.resolve(__dirname, '..');
const LogDir = path.join(Root, 'logs');
const Modules = { attention: '../inset_attention.js', moe: '../inset_moe.js', mixer: '../inset_mixer.js' };
const X0 = 20, Y0 = 50, W = 455;

const which = process.argv[2];
const only = process.argv[3] || '';
if (!Modules[which]) throw new Error('usage: node tests/preview_insets.js attention|moe|mixer [fixture-substring]');
const Inset = require(Modules[which]);

const panels = [];
for (const file of fs.readdirSync(path.join(__dirname, 'fixtures'))) {
  if (!file.includes(only)) continue;
  const spec = Arch.normalize(JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', file), 'utf8')).config);
  const kinds = [...new Set(spec.layerTypes.map(t => t.mixer).filter(m => m !== 'none'))];
  const jobs = [];
  if (which === 'attention' && kinds.some(k => ['full', 'sliding', 'chunked'].includes(k))) jobs.push({ kinds });
  if (which === 'moe' && spec.ffn.moe) jobs.push({});
  if (which === 'mixer' && spec.linearMixer) {
    for (const kind of kinds.filter(k => ['linear', 'mamba', 'conv'].includes(k))) jobs.push({ kind });
  }
  for (const extra of jobs) {
    const out = Inset.draw(spec, { x0: X0, y0: Y0, w: W, ...extra });
    const h = Math.ceil(out.y1 + 20);
    const anchors = [out.anchor, out.expertTop].filter(Boolean)
      .map(a => `<circle cx="${a.x}" cy="${a.y}" r="5" fill="#e00"/>`).join('');
    panels.push(`<figure style="margin:0 0 24px;display:inline-block;vertical-align:top">` +
      `<figcaption style="font:13px monospace">${file}</figcaption>` +
      `<svg xmlns="http://www.w3.org/2000/svg" width="${W + 40}" height="${h}" viewBox="0 0 ${W + 40} ${h}" style="border:1px solid #ccc">` +
      `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto">` +
      `<path d="M0 0 L10 5 L0 10 z" fill="#1a1a1a"/></marker><style>${Kit.Style}</style></defs>` +
      `${out.svg.join('')}${anchors}</svg></figure>`);
  }
}

fs.mkdirSync(LogDir, { recursive: true });
const html = path.join(LogDir, `preview_${which}.html`);
fs.writeFileSync(html, `<!doctype html><meta charset="utf-8"><body style="margin:8px;background:#fff;width:1560px">${panels.join('')}</body>`);
const chrome = ['ProgramFiles', 'ProgramFiles(x86)', 'LocalAppData']
  .map(v => process.env[v] && path.join(process.env[v], 'Google/Chrome/Application/chrome.exe'))
  .find(p => p && fs.existsSync(p));
const png = path.join(LogDir, `preview_${which}.png`);
execFileSync(chrome, ['--headless=new', '--disable-gpu', '--hide-scrollbars', '--window-size=1580,2600',
  `--screenshot=${png}`, 'file:///' + html.replace(/\\/g, '/')], { stdio: 'ignore' });
console.log(`${panels.length} panels -> ${path.relative(Root, png)}`);
