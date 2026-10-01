// Shared SVG drawing primitives, palette and base CSS for the figure and its insets.
// Classic script: the page loads it before diagram.js and the inset modules; node tests require() it.
(function (root) {
  const LH = 21;
  const BOX_H = 38;
  const PLUS_R = 12;

  const Palette = {
    ink: '#1a1a1a',
    dark: '#2b2b2b',
    mid: '#555555',
    gray: '#d2d2d2',
    light: '#e9e9e9',
    line: '#bdbdbd',
    white: '#ffffff',
    accent: '#bdb2ff',
    accentInk: '#5646d6',
    tint: '#efecff',
  };

  const MixerNames = {
    full: 'Full attention', sliding: 'Sliding window attention', chunked: 'Chunked attention',
    linear: 'Linear attention', mamba: 'Mamba', conv: 'Short convolution', none: 'No token mixer',
  };
  const FfnNames = { dense: 'Dense FFN', moe: 'MoE', none: 'No FFN' };
  const MixerColors = {
    full: '#2b2b2b', sliding: '#bdb2ff', chunked: '#7d6ff0',
    linear: '#0072b2', mamba: '#009e73', conv: '#cc79a7', none: '#ffffff',
  };
  const FfnColors = { dense: '#bdbdbd', moe: '#2b2b2b', none: '#ffffff' };
  // Label color on each mixer fill: light fills take ink, dark fills take white.
  const MixerInk = { full: '#ffffff', sliding: '#1a1a1a', chunked: '#ffffff', linear: '#ffffff', mamba: '#ffffff', conv: '#ffffff', none: '#1a1a1a' };

  const kindCss = Object.keys(MixerColors).map(k =>
    `.b-k-${k} { fill: ${MixerColors[k]}; stroke: #1a1a1a; stroke-width: 2; } ` +
    `.t-k-${k} { font-size: 15px; font-weight: 500; fill: ${MixerInk[k]}; } ` +
    // Lavender reads on the near-black box only; on the colored mixer fills highlights stay white.
    `.t-k-${k} .hl { fill: ${k === 'full' ? '#bdb2ff' : MixerInk[k] === '#ffffff' ? '#ffffff' : '#5646d6'}; font-weight: 700; }`).join('\n    ');

  const Style = `
    text { font-family: 'IBM Plex Sans KR', 'Helvetica Neue', Arial, sans-serif; fill: #1a1a1a; }
    .ln { fill: none; stroke: #1a1a1a; stroke-width: 2.2; }
    .ln-thin { fill: none; stroke: #1a1a1a; stroke-width: 1.2; }
    .ld { fill: none; stroke: #1a1a1a; stroke-width: 3; stroke-dasharray: 0.1 6.5; stroke-linecap: round; }
    .lda { fill: none; stroke: #5646d6; stroke-width: 3; stroke-dasharray: 0.1 6.5; stroke-linecap: round; }
    .outer { fill: #e4e4e4; stroke: #1a1a1a; stroke-width: 2.5; }
    .block { fill: #efecff; stroke: #1a1a1a; stroke-width: 2.5; }
    .b-plain { fill: #ffffff; stroke: #1a1a1a; stroke-width: 2; }
    .b-gray { fill: #d2d2d2; stroke: #1a1a1a; stroke-width: 2; }
    .b-dark { fill: #2b2b2b; stroke: #1a1a1a; stroke-width: 2; }
    .b-accent { fill: #ffffff; stroke: #5646d6; stroke-width: 2.6; }
    .b-router { fill: #ffffff; stroke: #5646d6; stroke-width: 2.6; }
    .b-lav { fill: #bdb2ff; stroke: #1a1a1a; stroke-width: 2; }
    .t-plain, .t-gray, .t-accent, .t-router, .t-lav { font-size: 15px; font-weight: 500; }
    .t-dark { font-size: 15px; font-weight: 500; fill: #ffffff; }
    .hl { fill: #5646d6; }
    .t-dark .hl { fill: #bdb2ff; font-weight: 700; }
    ${kindCss}
    .co { font-size: 17px; font-weight: 700; }
    .note { font-size: 14px; font-weight: 400; fill: #555555; }
    .small { font-size: 12px; font-weight: 500; fill: #555555; }
    .rep { font-size: 24px; font-weight: 700; fill: #5646d6; }
    .ttl { font-size: 18px; font-weight: 700; }
    .mono { font-family: 'IBM Plex Mono', Consolas, monospace; font-size: 16px; }
    .inset { fill: #e9e9e9; stroke: #1a1a1a; stroke-width: 3; stroke-dasharray: 0.1 7; stroke-linecap: round; }
    .inset-a { fill: #ffffff; stroke: #5646d6; stroke-width: 3; stroke-dasharray: 0.1 7; stroke-linecap: round; }
    .op { fill: #ffffff; stroke: #1a1a1a; stroke-width: 2; }
    .dot { fill: #1a1a1a; }
    .bd { fill: #1a1a1a; } .ba { fill: #5646d6; }
    .t-badge { font-size: 14px; font-weight: 700; fill: #ffffff; }
    .cell-w { stroke: #bdbdbd; stroke-width: 1; }
    .axis { font-size: 12px; fill: #555555; }
    .rowl { font-size: 13px; font-weight: 500; }`;

  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  // {text} marks a highlighted run inside a label
  const rich = s => esc(s).replace(/\{([^}]*)\}/g, '<tspan class="hl">$1</tspan>');
  const at = (y0, h) => ({ y0, h, y1: y0 + h, cy: y0 + h / 2 });
  // Widths stay exact (18,432); only token counts shorten to K/M (Arch.formatTokens).
  const fmtD = n => Number(n).toLocaleString('en-US');

  // Word wrap that keeps {highlight} runs whole; braces do not count toward the width.
  function wrap(str, max) {
    const words = str.match(/\{[^}]*\}\S*|\S+/g) || [];
    const lines = [];
    let cur = '';
    for (const w of words) {
      const next = cur ? `${cur} ${w}` : w;
      if (cur && next.replace(/[{}]/g, '').length > max) {
        lines.push(cur);
        cur = w;
      } else cur = next;
    }
    if (cur) lines.push(cur);
    return lines;
  }

  function text(x, y, lines, cls, anchor = 'start', lh = LH) {
    const spans = lines.map((ln, i) => `<tspan x="${x}" dy="${i ? lh : 0}">${rich(ln)}</tspan>`).join('');
    return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${spans}</text>`;
  }

  function box(cx, y0, w, h, lines, variant = 'plain') {
    const lh = 19;
    const base = y0 + h / 2 - ((lines.length - 1) * lh) / 2 + 5;
    return `<rect x="${cx - w / 2}" y="${y0}" width="${w}" height="${h}" rx="9" class="b-${variant}"/>` +
      text(cx, base, lines, `t-${variant}`, 'middle', lh);
  }

  const kindBox = (cx, y0, w, h, lines, kind) => box(cx, y0, w, h, lines, `k-${kind}`);

  const path = (pts, cls = 'ln', arrow = true) =>
    `<path d="M${pts.map(p => `${p[0]} ${p[1]}`).join(' L')}" class="${cls}"${arrow ? ' marker-end="url(#ah)"' : ''}/>`;

  const rect = (x, y, w, h, fill, extra = '') =>
    `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" fill="${fill}"${extra}/>`;

  function plus(cx, cy, r = PLUS_R) {
    return `<circle cx="${cx}" cy="${cy}" r="${r}" class="op"/>` +
      `<path d="M${cx - r + 5} ${cy} H${cx + r - 5} M${cx} ${cy - r + 5} V${cy + r - 5}" class="ln"/>`;
  }

  function times(cx, cy, r = 11) {
    const d = (r - 4) * 0.7071;
    return `<circle cx="${cx}" cy="${cy}" r="${r}" class="op"/>` +
      `<path d="M${cx - d} ${cy - d} L${cx + d} ${cy + d} M${cx - d} ${cy + d} L${cx + d} ${cy - d}" class="ln"/>`;
  }

  function badge(x, y, label, cls) {
    const w = Math.max(26, 12 + String(label).length * 9);
    return `<rect x="${x - w / 2}" y="${y - 13}" width="${w}" height="26" rx="5" class="${cls}"/>` +
      `<text x="${x}" y="${y + 5}" class="t-badge" text-anchor="middle">${esc(label)}</text>`;
  }

  function frame(x0, y0, w, h, title, accent = false) {
    return `<rect x="${x0}" y="${y0}" width="${w}" height="${h}" rx="22" class="${accent ? 'inset-a' : 'inset'}"/>` +
      (title ? text(x0, y0 - 14, [title], 'ttl') : '');
  }

  const api = {
    LH, BOX_H, PLUS_R, Palette, MixerNames, FfnNames, MixerColors, FfnColors, MixerInk, Style,
    esc, rich, at, fmtD, wrap, text, box, kindBox, path, rect, plus, times, badge, frame,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SvgKit = api;
})(this);
