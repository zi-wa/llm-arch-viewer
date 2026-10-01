// Sequence-mixer inset for the non-attention token mixers: Gated DeltaNet, Mamba / Mamba-2, short convolution.
(function (root) {
  const Kit = root.SvgKit || require('./svgkit.js');

  const BOX_W = 220;
  const GATE_W = 130;
  const GAP = 20;

  function skip(lines) { return lines.filter(Boolean); }

  // stages are listed top to bottom; a stage with `kind` gets that mixer's fill color
  function trunk(cx, y0, stages) {
    const s = [];
    let cur = Kit.at(y0 + 26, Kit.BOX_H);
    const boxes = [cur];
    for (let i = 1; i < stages.length; i++) {
      cur = Kit.at(cur.y1 + GAP, Kit.BOX_H);
      boxes.push(cur);
    }
    const insetBottom = boxes[boxes.length - 1].y1 + GAP;
    s.push(Kit.path([[cx, insetBottom], [cx, boxes[boxes.length - 1].y1]]));
    for (let i = boxes.length - 1; i > 0; i--) s.push(Kit.path([[cx, boxes[i].y0], [cx, boxes[i - 1].y1]]));
    stages.forEach((stage, i) => {
      s.push(stage.kind
        ? Kit.kindBox(cx, boxes[i].y0, BOX_W, Kit.BOX_H, stage.lines, stage.kind)
        : Kit.box(cx, boxes[i].y0, BOX_W, Kit.BOX_H, stage.lines));
    });
    return { svg: s, insetBottom, topY0: boxes[0].y0 };
  }

  function drawLinear(spec, x0, y0, w) {
    const lm = spec.linearMixer;
    const cx = x0 + w / 2;
    const stages = [
      { lines: ['Linear out'] },
      { lines: ['Output gate'] },
      { lines: skip(['Delta-rule state update', lm.heads != null && lm.headDim != null ? `state {${lm.heads}} x {${lm.headDim}} x {${lm.headDim}}` : null]), kind: 'linear' },
      { lines: ['L2-normalize q, k'] },
      { lines: skip(['Causal conv', lm.convKernel != null ? `kernel {${lm.convKernel}}` : null]) },
      { lines: ['Linear projections', 'to q, k, v, gates'] },
    ];
    const t = trunk(cx, y0, stages);
    return { svg: t.svg, y1: t.insetBottom, title: 'Gated DeltaNet module' };
  }

  function drawMamba(spec, x0, y0, w) {
    const lm = spec.linearMixer;
    const cx = x0 + w / 2 - 30;
    const rx = x0 + w - 85;
    const s = [];
    const ssmLines = skip(['Selective SSM',
      lm.stateSize != null ? `state size {${lm.stateSize}}` : null,
      lm.heads != null && lm.headDim != null ? `{${lm.heads}} heads x {${lm.headDim}}` : null]);
    const ssmH = ssmLines.length > 2 ? Kit.BOX_H + 16 : Kit.BOX_H;

    const top = Kit.at(y0 + 26, Kit.BOX_H);
    const mulY = top.y1 + 30;
    const ssm = Kit.at(mulY + 26, ssmH);
    const conv = Kit.at(ssm.y1 + GAP, Kit.BOX_H);
    const bottom = Kit.at(conv.y1 + GAP, Kit.BOX_H);
    const insetBottom = bottom.y1 + GAP;

    s.push(Kit.path([[cx, insetBottom], [cx, bottom.y1]]));
    s.push(Kit.path([[cx, bottom.y0], [cx, conv.y1]]));
    s.push(Kit.path([[cx, conv.y0], [cx, ssm.y1]]));
    s.push(Kit.path([[cx, ssm.y0], [cx, mulY + Kit.PLUS_R]]));
    s.push(Kit.path([[cx, mulY - Kit.PLUS_R], [cx, top.y1]]));
    s.push(Kit.times(cx, mulY));
    const gateY0 = mulY - Kit.BOX_H / 2;
    s.push(Kit.path([[cx, bottom.y0], [cx, bottom.y0 - 16], [rx, bottom.y0 - 16], [rx, gateY0 + Kit.BOX_H]]));
    s.push(Kit.path([[rx, gateY0], [cx + Kit.PLUS_R + 2, mulY]]));
    s.push(Kit.box(rx, gateY0, GATE_W, Kit.BOX_H, ['SiLU(z)']));

    s.push(Kit.box(cx, top.y0, BOX_W, Kit.BOX_H, ['Out-projection']));
    s.push(Kit.kindBox(cx, ssm.y0, BOX_W, ssmH, ssmLines, 'mamba'));
    s.push(Kit.box(cx, conv.y0, BOX_W, Kit.BOX_H, skip(['Conv1d', lm.convKernel != null ? `kernel {${lm.convKernel}}` : null])));
    s.push(Kit.box(cx, bottom.y0, BOX_W, Kit.BOX_H, ['In-projection', 'splits x, z']));

    const title = lm.kind === 'Mamba-2' ? 'Mamba-2 module' : 'Mamba module';
    return { svg: s, y1: insetBottom, title };
  }

  function drawConv(spec, x0, y0, w) {
    const lm = spec.linearMixer;
    const cx = x0 + w / 2;
    const s = [];
    const top = Kit.at(y0 + 26, Kit.BOX_H);
    const mulY2 = top.y1 + 30;
    const conv = Kit.at(mulY2 + 26, Kit.BOX_H);
    const mulY1 = conv.y1 + 30;
    const bottom = Kit.at(mulY1 + 26, Kit.BOX_H);
    const insetBottom = bottom.y1 + GAP;

    s.push(Kit.path([[cx, insetBottom], [cx, bottom.y1]]));
    s.push(Kit.path([[cx, bottom.y0], [cx, mulY1 + Kit.PLUS_R]]));
    s.push(Kit.times(cx, mulY1));
    s.push(Kit.path([[cx, mulY1 - Kit.PLUS_R], [cx, conv.y1]]));
    s.push(Kit.path([[cx, conv.y0], [cx, mulY2 + Kit.PLUS_R]]));
    s.push(Kit.times(cx, mulY2));
    s.push(Kit.path([[cx, mulY2 - Kit.PLUS_R], [cx, top.y1]]));

    const rx = x0 + w - 66;
    const rx2 = x0 + w - 36;
    s.push(Kit.path([[cx, bottom.y0], [cx, bottom.y0 - 14], [rx, bottom.y0 - 14], [rx, mulY1], [cx + Kit.PLUS_R + 2, mulY1]]));
    s.push(Kit.text(rx - 6, bottom.y0 - 20, ['B'], 'small', 'middle'));
    s.push(Kit.path([[cx, bottom.y0], [cx, bottom.y0 - 26], [rx2, bottom.y0 - 26], [rx2, mulY2], [cx + Kit.PLUS_R + 2, mulY2]]));
    s.push(Kit.text(rx2 + 14, bottom.y0 - 32, ['C'], 'small', 'middle'));

    s.push(Kit.box(cx, top.y0, BOX_W, Kit.BOX_H, ['Out-projection']));
    s.push(Kit.kindBox(cx, conv.y0, BOX_W, Kit.BOX_H, skip(['Depthwise short conv', lm.convKernel != null ? `kernel {${lm.convKernel}}` : null]), 'conv'));
    s.push(Kit.box(cx, bottom.y0, BOX_W, Kit.BOX_H, ['In-projection', 'splits B, C, x']));

    return { svg: s, y1: insetBottom, title: 'Short convolution module' };
  }

  function drawGeneric(spec, x0, y0, w, kind) {
    const label = (spec.linearMixer && spec.linearMixer.kind) || kind;
    const cx = x0 + w / 2;
    const t = trunk(cx, y0, [
      { lines: ['Output'] },
      { lines: [`{${label}}`] },
      { lines: ['Input'] },
    ]);
    return { svg: t.svg, y1: t.insetBottom, title: `${label} module` };
  }

  /**
   * @param {import('./arch.js').Spec} spec  spec.linearMixer is non-null
   * @param {{x0:number, y0:number, w:number, kind:'linear'|'mamba'|'conv'}} opts
   * @returns {{svg:string[], y1:number, anchor:{x:number,y:number}}}
   */
  function draw(spec, opts) {
    const { x0, y0, w, kind } = opts;
    let out;
    if (kind === 'linear') out = drawLinear(spec, x0, y0, w);
    else if (kind === 'mamba') out = drawMamba(spec, x0, y0, w);
    else if (kind === 'conv') out = drawConv(spec, x0, y0, w);
    else out = drawGeneric(spec, x0, y0, w, kind);

    const h = out.y1 - y0;
    const svg = [Kit.frame(x0, y0, w, h, out.title, true), ...out.svg];
    return { svg, y1: out.y1, anchor: { x: x0, y: y0 + h / 2 } };
  }


  const api = { draw };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MixerInset = api;
})(this);
