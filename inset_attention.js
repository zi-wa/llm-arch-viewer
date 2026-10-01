// Attention inset: how this model's attention is shaped - head grouping (MHA/GQA/MQA) or the MLA latent path,
// plus one attention-mask thumbnail per attention kind the model uses (global, sliding window, chunked, sinks).
(function (root) {
  const Kit = root.SvgKit || require('./svgkit.js');

  const PAD = 20;
  const MIN_CELL_W = 2;
  const MAX_CELL_W = 16;
  const CELL_GAP = 1;
  const GROUP_GAP = 10;
  const ELLIPSIS_W = 14;
  const GROUP_KEEP = [3, 2]; // groups drawn (first, last) when the group count does not fit
  const HEAD_KEEP = [5, 2]; // heads-per-group drawn (first, last) when the group itself does not fit

  const KIND_TITLE = {
    MHA: 'Multi-head attention',
    GQA: 'Grouped-query attention',
    MQA: 'Multi-query attention',
    MLA: 'Multi-head latent attention',
  };

  const MASK_WORD = { full: 'global', sliding: 'local', chunked: 'chunked' };
  const MASK_ORDER = ['sliding', 'chunked', 'full'];

  const range = (n) => Array.from({ length: n }, (_, i) => i);

  // indices to draw for an axis of n items when it does not fit whole; -1 marks the ellipsis gap
  function pickIndices(n, keep) {
    const [head, tail] = keep;
    const slots = [];
    for (let i = 0; i < head; i++) slots.push(i);
    slots.push(-1);
    for (let i = n - tail; i < n; i++) slots.push(i);
    return slots;
  }

  // smallest p with seq[i] === seq[i + p] wherever both indices exist; seq.length when the sequence never repeats
  function detectPeriod(seq) {
    const n = seq.length;
    for (let p = 1; p < n; p++) {
      let matches = true;
      for (let i = 0; i + p < n; i++) {
        if (seq[i] !== seq[i + p]) { matches = false; break; }
      }
      if (matches) return p;
    }
    return n;
  }

  // '{5} local : {1} global per {6} layers', or the plain-count fallback '{52} local, {10} global'
  function maskRatioLabel(spec, attnKinds) {
    if (attnKinds.length < 2) return null;
    const seq = spec.layerTypes.map((lt) => lt.mixer).filter((m) => attnKinds.indexOf(m) !== -1);
    if (seq.length < 2) return null;
    const order = MASK_ORDER.filter((k) => attnKinds.indexOf(k) !== -1);
    const period = detectPeriod(seq);
    const periodic = period < seq.length;
    const source = periodic ? seq.slice(0, period) : seq;
    const counts = {};
    for (const k of source) counts[k] = (counts[k] || 0) + 1;
    const parts = order.filter((k) => counts[k]).map((k) => `{${counts[k]}} ${MASK_WORD[k]}`);
    return periodic ? `${parts.join(' : ')} per {${period}} layers` : parts.join(', ');
  }

  // query heads on top, each group fanning into the KV head it shares
  function drawHeadPanel(spec, x0, y0, w) {
    const a = spec.attention;
    const heads = a.heads;
    const kvHeads = Math.max(1, a.kvHeads);
    const groupSize = Math.max(1, Math.round(heads / kvHeads));
    const usableW = w - 2 * PAD;

    const fitsFull = kvHeads * groupSize * MIN_CELL_W + kvHeads * Math.max(0, groupSize - 1) * CELL_GAP +
      Math.max(0, kvHeads - 1) * GROUP_GAP <= usableW;
    const groupSlots = fitsFull || kvHeads <= GROUP_KEEP[0] + GROUP_KEEP[1] ? range(kvHeads) : pickIndices(kvHeads, GROUP_KEEP);
    const headSlots = fitsFull || groupSize <= HEAD_KEEP[0] + HEAD_KEEP[1] ? range(groupSize) : pickIndices(groupSize, HEAD_KEEP);

    const drawnGroups = groupSlots.filter((g) => g !== -1).length;
    const groupHasEllipsis = groupSlots.indexOf(-1) !== -1;
    const drawnHeadsPerGroup = headSlots.filter((h) => h !== -1).length;
    const headHasEllipsis = headSlots.indexOf(-1) !== -1;

    const groupFixedW = Math.max(0, drawnHeadsPerGroup - 1) * CELL_GAP + (headHasEllipsis ? ELLIPSIS_W + CELL_GAP : 0);
    const groupSlotCount = drawnGroups + (groupHasEllipsis ? 1 : 0);
    const totalFixed = drawnGroups * groupFixedW + Math.max(0, groupSlotCount - 1) * GROUP_GAP +
      (groupHasEllipsis ? ELLIPSIS_W : 0);
    const totalCells = drawnGroups * drawnHeadsPerGroup;
    let cellW = (usableW - totalFixed) / totalCells;
    if (!isFinite(cellW) || cellW < MIN_CELL_W) cellW = MIN_CELL_W;
    if (cellW > MAX_CELL_W) cellW = MAX_CELL_W;

    const qH = 24, kvH = 28, rowGap = 30;
    const qY = y0, kvY = qY + qH + rowGap;
    const svg = [];
    let x = x0 + PAD;

    groupSlots.forEach((g, gi) => {
      if (g === -1) {
        svg.push(Kit.text(x + ELLIPSIS_W / 2, (qY + qH + kvY) / 2, ['...'], 'note', 'middle'));
        x += ELLIPSIS_W;
      } else {
        const groupStartX = x;
        headSlots.forEach((h, hi) => {
          if (h === -1) {
            svg.push(Kit.text(x + ELLIPSIS_W / 2, qY + qH / 2 + 5, ['...'], 'small', 'middle'));
            x += ELLIPSIS_W;
          } else {
            svg.push(Kit.rect(x, qY, cellW, qH, Kit.Palette.gray, ' stroke="#1a1a1a" stroke-width="1"'));
            x += cellW;
          }
          if (hi < headSlots.length - 1) x += CELL_GAP;
        });
        const groupEndX = x;
        const kvCx = (groupStartX + groupEndX) / 2;
        const kvW = Math.max(cellW, groupEndX - groupStartX);
        svg.push(Kit.rect(kvCx - kvW / 2, kvY, kvW, kvH, Kit.Palette.dark, ' stroke="#1a1a1a" stroke-width="1"'));
        let lx = groupStartX;
        headSlots.forEach((h) => {
          if (h !== -1) {
            svg.push(Kit.path([[lx + cellW / 2, qY + qH], [kvCx, kvY]], 'ln-thin', false));
            lx += cellW;
          } else {
            lx += ELLIPSIS_W;
          }
          lx += CELL_GAP;
        });
      }
      if (gi < groupSlots.length - 1) x += GROUP_GAP;
    });

    const headWord = heads === 1 ? 'head' : 'heads';
    const kvWord = kvHeads === 1 ? 'head' : 'heads';
    const queryWord = groupSize === 1 ? 'query' : 'queries';
    const labelY = kvY + kvH + 26;
    svg.push(Kit.text(x0 + PAD, labelY, [`{${Kit.fmtD(heads)}} query ${headWord}`], 'note'));
    svg.push(Kit.text(x0 + PAD, labelY + 20, [`{${Kit.fmtD(kvHeads)}} KV ${kvWord}, {${Kit.fmtD(groupSize)}} ${queryWord} each`], 'note'));
    svg.push(Kit.text(x0 + PAD, labelY + 40, [`head dim {${Kit.fmtD(a.headDim)}}`], 'note'));

    return { svg, bottom: labelY + 40 + 6, labelTop: labelY - 14, anchor: { x: x0, y: qY + qH / 2 } };
  }

  // bar lengths are proportional to the widths; the accent bars are what MLA keeps in the KV cache
  function drawMlaPanel(spec, x0, y0, w) {
    const a = spec.attention;
    const mla = a.mla;
    const hidden = spec.hidden;
    const usableW = w - 2 * PAD;
    const labelRoom = 210;
    const scale = Math.max(1, usableW - labelRoom) / hidden;
    const barH = 20, rowGap = 32;
    const barX = x0 + PAD;
    const svg = [];
    let y = y0;

    function bar(value, fill, lineText) {
      const bw = Math.max(4, value * scale);
      svg.push(Kit.rect(barX, y, bw, barH, fill, ' stroke="#1a1a1a" stroke-width="1"'));
      svg.push(Kit.text(barX + bw + 8, y + barH - 5, [lineText], 'note'));
      y += rowGap;
    }

    const mid = Kit.Palette.mid;
    const cached = Kit.Palette.accentInk;
    const qkHeadDim = mla.qkNopeHeadDim + mla.qkRopeHeadDim;

    bar(hidden, mid, `hidden {${Kit.fmtD(hidden)}}`);
    if (mla.qLoraRank) bar(mla.qLoraRank, mid, `query down-projection -> q latent {${Kit.fmtD(mla.qLoraRank)}}`);
    bar(qkHeadDim, mid, `up-projection -> per-head Q dim {${Kit.fmtD(qkHeadDim)}} x {${Kit.fmtD(a.heads)}} heads`);
    bar(mla.kvLoraRank, cached, `down-projection -> latent KV {${Kit.fmtD(mla.kvLoraRank)}} (cached)`);
    bar(mla.qkRopeHeadDim, cached, `decoupled RoPE key {${Kit.fmtD(mla.qkRopeHeadDim)}} (cached)`);
    bar(qkHeadDim, mid, `up-projection -> per-head K dim {${Kit.fmtD(qkHeadDim)}} x {${Kit.fmtD(a.heads)}} heads`);
    bar(mla.vHeadDim, mid, `up-projection -> per-head V dim {${Kit.fmtD(mla.vHeadDim)}} x {${Kit.fmtD(a.heads)}} heads`);

    const mlaCache = mla.kvLoraRank + mla.qkRopeHeadDim;
    // Full per-head keys and values at these widths; spec.attention.headDim is hidden / heads for MLA configs.
    const mhaEquivalent = a.heads * (qkHeadDim + mla.vHeadDim);
    svg.push(Kit.text(barX, y + 14, [
      `cached per token: {${Kit.fmtD(mlaCache)}} values (latent + RoPE key)`,
      `MHA with the same heads: ${Kit.fmtD(mhaEquivalent)} values`,
    ], 'note', 'start', 19));
    y += rowGap + 19;

    return { svg, bottom: y, anchor: { x: x0, y: y0 + barH / 2 } };
  }

  // `right` puts the masks next to the head labels instead of below them
  function drawMaskPanel(spec, x0, y0, w, attnKinds, right) {
    const N = 12, cell = right ? 7 : 9, gap = 26;
    const maskW = N * cell;
    const order = MASK_ORDER.filter((k) => attnKinds.indexOf(k) !== -1);
    const titleY = y0;
    const gridY = y0 + 8;
    const svg = [];
    let x = right ? x0 + w - PAD - order.length * maskW - (order.length - 1) * gap : x0 + PAD;

    for (const kind of order) {
      for (let row = 0; row < N; row++) {
        for (let col = 0; col < N; col++) {
          let filled = col <= row;
          if (kind === 'sliding') filled = filled && row - col < 3;
          if (kind === 'chunked') filled = filled && Math.floor(row / 4) === Math.floor(col / 4);
          const sink = spec.attention.sinks && col === 0;
          if (filled || sink) {
            const fill = sink ? Kit.Palette.accentInk : Kit.MixerColors[kind];
            svg.push(Kit.rect(x + col * cell, gridY + row * cell, cell - 1, cell - 1, fill));
          }
        }
      }
      svg.push(`<rect x="${x}" y="${gridY}" width="${maskW}" height="${maskW}" fill="none" stroke="#1a1a1a" stroke-width="1.5"/>`);
      svg.push(Kit.text(x, titleY, [MASK_WORD[kind]], 'small'));
      const detail = kind === 'sliding' ? `window {${Kit.fmtD(spec.attention.slidingWindow)}}`
        : kind === 'chunked' ? `chunk {${Kit.fmtD(spec.attention.chunkSize)}}` : null;
      if (detail) svg.push(Kit.text(x, gridY + maskW + 15, [detail], 'small'));
      x += maskW + gap;
    }

    const bottom = gridY + maskW + (order.some(k => k !== 'full') ? 20 : 4);
    return { svg, bottom, ratio: maskRatioLabel(spec, attnKinds) };
  }

  /**
   * @param {import('./arch.js').Spec} spec
   * @param {{x0:number, y0:number, w:number, kinds:string[]}} opts
   *   kinds: mixer kinds present in spec.layerTypes (subset of 'full','sliding','chunked','linear','mamba','conv')
   * @returns {{svg:string[], y1:number, anchor:{x:number,y:number}}}
   *   everything drawn inside x0..x0+w from y0 (title sits above y0 via Kit.frame); y1 = lowest drawn y;
   *   anchor = point on the frame's left edge for the main figure's dotted leader
   */
  function draw(spec, opts) {
    const { x0, y0, w } = opts;
    const attnKinds = (opts.kinds || []).filter((k) => k === 'full' || k === 'sliding' || k === 'chunked');
    const kind = spec.attention.kind;
    const title = KIND_TITLE[kind] || KIND_TITLE.MHA;

    const contentY0 = y0 + 16;
    const panel = kind === 'MLA' ? drawMlaPanel(spec, x0, contentY0, w) : drawHeadPanel(spec, x0, contentY0, w);
    const svg = panel.svg.slice();

    let bottom = panel.bottom;
    if (attnKinds.length > 0) {
      // MLA bar labels use the full width, so its masks still go below
      const beside = kind !== 'MLA';
      const mask = drawMaskPanel(spec, x0, beside ? panel.labelTop : bottom + 24, w, attnKinds, beside);
      svg.push(...mask.svg);
      bottom = Math.max(bottom, mask.bottom);
      if (mask.ratio) {
        svg.push(Kit.text(x0 + PAD, bottom + 20, [mask.ratio], 'note'));
        bottom += 26;
      }
    }

    const h = bottom - y0 + 16;
    const y1 = y0 + h;
    return { svg: [Kit.frame(x0, y0, w, h, title), ...svg], y1, anchor: panel.anchor };
  }


  const api = { draw };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AttentionInset = api;
})(this);
