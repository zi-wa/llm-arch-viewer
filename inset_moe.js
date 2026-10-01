// MoE inset: the router and every expert drawn as its own cell, so 8 experts and 512 experts look different;
// the top-k active per token and the shared experts are marked.
(function (root) {
  const Kit = root.SvgKit || require('./svgkit.js');
  const Arch = root.Arch || require('./arch.js');

  const PAD = 18;
  const CELL_GAP = 2;
  const CELL_MIN = 6;
  const CELL_MAX = 24;
  const MAX_GRID_H = 170;
  const ROUTER_W = 120;
  const ROUTER_H = 40;
  const SHARED_COL_W = 64;

  // largest cell that still fits MAX_GRID_H; few experts stay a tight cluster instead of a long row
  function computeGrid(experts, availW) {
    for (let cell = CELL_MAX; cell >= CELL_MIN; cell--) {
      const cols = Math.max(1, Math.min(experts, Math.floor((availW + CELL_GAP) / (cell + CELL_GAP))));
      const rows = Math.ceil(experts / cols);
      if (rows * (cell + CELL_GAP) - CELL_GAP <= MAX_GRID_H) return { cell, cols, rows };
    }
    const cell = CELL_MIN;
    const cols = Math.max(1, Math.min(experts, Math.floor((availW + CELL_GAP) / (cell + CELL_GAP))));
    return { cell, cols, rows: Math.ceil(experts / cols) };
  }

  // Golden-ratio scatter over [0, experts): an even stride lines up in one grid column whenever the stride
  // is a multiple of the column count (GLM-4.5-Air: 128 experts, top-8, 16 columns).
  function activeIndices(experts, topK) {
    const set = new Set();
    for (let i = 0; set.size < Math.min(topK, experts); i++) {
      let idx = Math.floor(((i * 0.6180339887 + 0.37) % 1) * experts);
      while (set.has(idx)) idx = (idx + 1) % experts;
      set.add(idx);
    }
    return set;
  }

  /**
   * @param {import('./arch.js').Spec} spec  spec.ffn.moe is non-null
   * @param {{x0:number, y0:number, w:number}} opts
   * @returns {{svg:string[], y1:number, anchor:{x:number,y:number}, expertTop:{x:number,y:number}}}
   *   anchor = left-edge point for the main figure's accent leader from the MoE box;
   *   expertTop = top of one routed expert cell, where the FeedForward module's leader lands
   */
  function draw(spec, opts) {
    const { x0, y0, w } = opts;
    const m = spec.ffn.moe;
    const experts = m.experts;
    const topK = m.topK;
    const shared = m.sharedExperts || 0;

    const innerX0 = x0 + PAD;
    const innerX1 = x0 + w - PAD;
    const sharedGap = shared > 0 ? 16 : 0;
    const sharedColW = shared > 0 ? SHARED_COL_W : 0;
    const gridAvailW = innerX1 - innerX0 - sharedColW - sharedGap;
    const { cell, cols, rows } = computeGrid(experts, gridAvailW);
    const gridW = cols * cell + (cols - 1) * CELL_GAP;
    const gridH = rows * cell + (rows - 1) * CELL_GAP;
    const gridX0 = innerX0 + (gridAvailW - gridW) / 2;

    const routerCx = x0 + w / 2;
    const sumY = y0 + 30;
    const gridTop = sumY + Kit.PLUS_R + 34;
    const gridBottom = gridTop + gridH;
    const routerY0 = gridBottom + 34;
    const routerY1 = routerY0 + ROUTER_H;

    const s = [];

    s.push(Kit.path([[routerCx, sumY - Kit.PLUS_R], [routerCx, y0 + 14]]));
    const active = activeIndices(experts, topK);
    const gridLeftX = gridX0 + cell / 2;
    const gridMidX = gridX0 + gridW / 2;
    const gridRightX = gridX0 + gridW - cell / 2;

    for (const gx of [gridLeftX, gridMidX, gridRightX]) {
      s.push(Kit.path([[gx, gridTop - 2], [routerCx, sumY + Kit.PLUS_R + 16]]));
    }
    s.push(Kit.path([[routerCx, sumY + Kit.PLUS_R + 16], [routerCx, sumY + Kit.PLUS_R]]));
    s.push(Kit.plus(routerCx, sumY));

    for (const gx of [gridLeftX, gridMidX, gridRightX]) {
      s.push(Kit.path([[routerCx, gridBottom + 16], [gx, gridBottom + 2]]));
    }
    s.push(Kit.path([[routerCx, routerY0], [routerCx, gridBottom + 16]], 'ln', false));

    for (let i = 0; i < experts; i++) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      const x = gridX0 + col * (cell + CELL_GAP);
      const y = gridTop + row * (cell + CELL_GAP);
      const on = active.has(i);
      s.push(Kit.rect(x, y, cell, cell, on ? Kit.Palette.accentInk : Kit.Palette.light,
        ` rx="2" stroke="${on ? Kit.Palette.accentInk : Kit.Palette.line}" stroke-width="1"`));
    }
    const firstActive = Math.min(...active);
    const faRow = Math.floor(firstActive / cols);
    const faCol = firstActive % cols;
    const expertTop = { x: gridX0 + faCol * (cell + CELL_GAP) + cell / 2, y: gridTop + faRow * (cell + CELL_GAP) };

    if (shared > 0) {
      const sharedCell = Math.min(34, Math.max(22, cell + 10));
      const sharedCx = innerX1 - sharedColW / 2;
      const sharedCy = gridTop + gridH / 2;
      const sharedY = sharedCy - sharedCell / 2;
      s.push(Kit.path([[sharedCx, sharedY], [sharedCx, sumY], [routerCx + Kit.PLUS_R + 2, sumY]]));
      s.push(Kit.rect(sharedCx - sharedCell / 2, sharedY, sharedCell, sharedCell, Kit.Palette.dark, ' rx="3"'));
      if (shared > 1) s.push(Kit.badge(sharedCx, sharedY - 14, shared, 'ba'));
      s.push(Kit.text(sharedCx, sharedY + sharedCell + 17, ['shared'], 'small', 'middle'));
    }

    s.push(Kit.box(routerCx, routerY0, ROUTER_W, ROUTER_H, ['Router'], 'router'));

    const fmtD = Kit.fmtD;
    const fmtC = Arch.formatCount;
    // expert width is already shown by the bars next to the FFN module
    const facts = [
      `{${fmtD(experts)}} routed${shared ? ` + {${fmtD(shared)}} shared` : ''} expert${experts + shared > 1 ? 's' : ''}`,
      `{${fmtD(topK)}} active per token, {${m.router}} router`,
    ];
    if (spec.params && spec.params.total) {
      facts.push(`{${fmtC(spec.params.active)}} of {${fmtC(spec.params.total)}} parameters active`);
    }

    let fy = routerY1 + 32;
    const maxChars = 40;
    for (const fact of facts) {
      const lines = Kit.wrap(fact, maxChars);
      s.push(`<circle cx="${innerX0 + 4}" cy="${fy - 5}" r="3.2" class="dot"/>`);
      s.push(Kit.text(innerX0 + 16, fy, lines, 'note'));
      fy += Kit.LH * lines.length + 6;
    }

    const y1 = fy - 6 + PAD;
    s.unshift(Kit.frame(x0, y0, w, y1 - y0, 'MoE layer', true));

    return {
      svg: s,
      y1,
      anchor: { x: x0, y: gridTop + gridH / 2 },
      expertTop,
    };
  }


  const api = { draw };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MoeInset = api;
})(this);
