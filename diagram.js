// Builds the SVG figure from an Arch spec. Layout follows Raschka's LLM Architecture Gallery:
// block stack on the left, detail insets on the right.
(function (root) {
  const Kit = root.SvgKit || require('./svgkit.js');
  const { LH, BOX_H, PLUS_R, Palette, MixerNames, FfnNames, MixerColors, FfnColors } = Kit;
  const { esc, at, fmtD, wrap, text, box, kindBox, path, rect, plus, times, frame } = Kit;

  const W = 1740;
  const CX = 450;
  const BOX_W = 220, GAP = 28;
  const MIX_W = 260, SLOT_W = 310, VARIANT_GAP = 10;
  const OUTER_X0 = 200, OUTER_X1 = 680;
  const BLOCK_X0 = 272, BLOCK_X1 = 640;
  const XRES = 618;
  const SIDE_X = 96, SIDE_W = 140;
  const RX_A = 760, RX_B = 1265, RW = 455;
  const LEFT_CHARS = 19, RIGHT_CHARS = 42;
  const FORK = 8;

  const AttnKinds = new Set(['full', 'sliding', 'chunked']);
  const SeqKinds = new Set(['linear', 'mamba', 'conv']);
  // Attention kinds sit left of sequence mixers so the RoPE and QK-Norm arrows land on an attention box.
  const KindOrder = ['sliding', 'chunked', 'full', 'linear', 'mamba', 'conv'];

  const unique = arr => [...new Set(arr)];
  const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

  // Smallest repeating period of a layer sequence (0 when it does not repeat).
  function period(seq) {
    for (let p = 1; p <= Math.min(24, seq.length / 2); p++) {
      if (seq.every((v, i) => v === seq[i % p])) return p;
    }
    return 0;
  }

  function ffnName(spec) {
    const act = String(spec.ffn.activation || '').toLowerCase();
    const stem = /silu|swish|swiglu/.test(act) ? 'Swi' : /gelu/.test(act) ? 'Ge' : /relu/.test(act) ? 'Re' : '';
    if (spec.ffn.gated) return `${stem}GLU`;
    if (/gelu/.test(act)) return 'GELU';
    if (/relu/.test(act)) return 'ReLU';
    return act || 'MLP';
  }

  function activationLabel(spec) {
    const act = String(spec.ffn.activation || '').toLowerCase();
    if (/silu|swish|swiglu/.test(act)) return 'SiLU activation';
    if (/gelu/.test(act)) return 'GELU activation';
    if (/relu/.test(act)) return 'ReLU activation';
    return act ? `${act} activation` : 'Activation';
  }

  function mixerName(kind, spec) {
    if (SeqKinds.has(kind) && spec.linearMixer && spec.linearMixer.kind) return spec.linearMixer.kind;
    return MixerNames[kind] || kind;
  }

  function attnLines(spec) {
    const a = spec.attention;
    if (a.kind === 'MLA') return ['Multi-head {latent}', 'attention'];
    const kind = { GQA: 'grouped-query', MQA: 'multi-query' }[a.kind] || 'multi-head';
    return spec.family === 'encoder' ? [cap(kind), 'attention'] : [`Masked ${kind}`, 'attention'];
  }

  function kindLines(kind, spec, alone) {
    const a = spec.attention.kind;
    if (kind === 'full') return alone ? attnLines(spec) : ['Global', a];
    if (kind === 'sliding') return ['Sliding window', a];
    if (kind === 'chunked') return ['Chunked', a];
    const words = mixerName(kind, spec).split(' ');
    const lines = words.length > 1 ? [words[0], words.slice(1).join(' ')] : words;
    return alone && lines.length === 1 ? [lines[0], kind === 'conv' ? 'block' : 'mixer'] : lines;
  }

  function denseNote(spec, ffnSeq) {
    const dense = [];
    const sparse = [];
    ffnSeq.forEach((f, i) => (f === 'dense' ? dense : f === 'moe' ? sparse : []).push(i));
    if (!dense.length || !sparse.length) return null;
    const size = spec.ffn.intermediate ? ` with hidden size {${fmtD(spec.ffn.intermediate)}}` : '';
    if (dense.every((d, i) => d === i)) return `First {${dense.length}} blocks use dense FFN${size} instead of MoE`;
    const p = period(ffnSeq);
    if (p) {
      const k = ffnSeq.slice(0, p).filter(f => f === 'moe').length;
      return `MoE in {${k}} of every {${p}} blocks, the others use dense FFN${size}`;
    }
    return `{${dense.length}} of {${ffnSeq.length}} blocks use dense FFN${size}`;
  }

  function ffnInset(x0, y0, w, spec) {
    const s = [];
    if (spec.ffn.gated) {
      const lx = x0 + 128, rx = x0 + w - 110;
      const top = at(y0 + 26, BOX_H);
      const mulY = top.y1 + 26;
      const mid = at(mulY + 26, BOX_H);
      const bottom = at(mid.y1 + 34, BOX_H);
      const insetBottom = bottom.y1 + 34;
      const split = insetBottom - 14;
      const mx = (lx + rx) / 2;
      s.push(path([[lx, insetBottom], [lx, bottom.y1]]));
      s.push(path([[lx, split], [rx, split], [rx, mid.y1]]));
      s.push(path([[lx, bottom.y0], [lx, mid.y1]]));
      s.push(path([[lx, mid.y0], [lx, top.y1]]));
      s.push(path([[rx, mid.y0], [rx, mulY], [mx + 11, mulY]], 'ln', false));
      s.push(path([[mx - 11, mulY], [lx, mulY]], 'ln', false));
      s.push(times(mx, mulY));
      s.push(box(lx, top.y0, 180, BOX_H, ['Linear layer']));
      s.push(box(lx, mid.y0, 180, BOX_H, [activationLabel(spec)]));
      s.push(box(rx, mid.y0, 160, BOX_H, ['Linear layer']));
      s.push(box(lx, bottom.y0, 180, BOX_H, ['Linear layer']));
      return { svg: s, y1: insetBottom };
    }
    const cx = x0 + w / 2;
    const top = at(y0 + 26, BOX_H);
    const mid = at(top.y1 + 30, BOX_H);
    const bottom = at(mid.y1 + 30, BOX_H);
    const insetBottom = bottom.y1 + 30;
    s.push(path([[cx, insetBottom], [cx, bottom.y1]]));
    s.push(path([[cx, bottom.y0], [cx, mid.y1]]));
    s.push(path([[cx, mid.y0], [cx, top.y1]]));
    s.push(box(cx, top.y0, 180, BOX_H, ['Linear layer']));
    s.push(box(cx, mid.y0, 180, BOX_H, [activationLabel(spec)]));
    s.push(box(cx, bottom.y0, 180, BOX_H, ['Linear layer']));
    return { svg: s, y1: insetBottom };
  }

  // Embedding width against the FFN (or expert) width as bars: dense models widen, fine-grained experts narrow.
  function widthBars(x0, y0, spec, dim, moe) {
    const s = [];
    const rows = [['embedding dim', spec.hidden, Palette.dark], [moe ? 'expert hidden dim' : 'intermediate dim', dim, Palette.accentInk]];
    const scale = 130 / Math.max(spec.hidden, dim);
    rows.forEach(([label, v, fill], i) => {
      const y = y0 + i * 26;
      s.push(rect(x0, y, Math.max(2, v * scale), 14, fill));
      s.push(text(x0 + 8 + v * scale, y + 12, [`${fmtD(v)} ${label}`], 'note'));
    });
    const ratio = dim / spec.hidden;
    s.push(text(x0, y0 + 70, [`FFN width = {${ratio >= 1 ? ratio.toFixed(2) : ratio.toFixed(3)}} x embedding dim`], 'co'));
    return { svg: s, y1: y0 + 76 };
  }

  // Box of one FFN kind; MoE gets stacked outlines behind it to read as many parallel experts.
  function ffnKindBox(cx, y0, w, h, lines, kind) {
    const stack = kind === 'moe'
      ? [10, 5].map(d => `<rect x="${cx - w / 2 + d}" y="${y0 - d}" width="${w}" height="${h}" rx="9" class="b-gray"/>`).join('')
      : '';
    return stack + box(cx, y0, w, h, lines, 'gray');
  }

  function layoutVariants(slot) {
    const n = slot.variants.length;
    const bw = (slot.w - VARIANT_GAP * (n - 1)) / n;
    slot.variants.forEach((v, i) => {
      v.w = bw;
      v.cx = CX - slot.w / 2 + bw / 2 + i * (bw + VARIANT_GAP);
    });
  }

  // Arrow from the item below into a slot; a slot with several boxes gets a fork.
  function linkUp(lower, upper) {
    const out = [];
    const into = upper.variants && upper.variants.length > 1;
    const from = lower.variants && lower.variants.length > 1;
    let y0 = lower.y0;
    if (from) {
      const joinY = lower.y0 - FORK;
      for (const v of lower.variants) out.push(path([[v.cx, lower.y0], [v.cx, joinY]], 'ln', false));
      out.push(path([[lower.variants[0].cx, joinY], [lower.variants[lower.variants.length - 1].cx, joinY]], 'ln', false));
      y0 = joinY;
    }
    if (!into) {
      out.push(path([[CX, y0], [CX, upper.y1]]));
      return out;
    }
    const forkY = upper.y1 + FORK;
    out.push(path([[CX, y0], [CX, forkY]], 'ln', false));
    out.push(path([[upper.variants[0].cx, forkY], [upper.variants[upper.variants.length - 1].cx, forkY]], 'ln', false));
    for (const v of upper.variants) out.push(path([[v.cx, forkY], [v.cx, upper.y1]]));
    return out;
  }

  function render(spec) {
    const arch = root.Arch || require('./arch.js');
    const Insets = {
      attention: root.AttentionInset || require('./inset_attention.js'),
      moe: root.MoeInset || require('./inset_moe.js'),
      mixer: root.MixerInset || require('./inset_mixer.js'),
    };
    const fmtC = arch.formatCount;
    const fmtT = arch.formatTokens;
    const types = spec.layerTypes || [];
    const mixSeq = types.map(t => t.mixer);
    const ffnSeq = types.map(t => t.ffn);
    const mixKinds = KindOrder.filter(k => mixSeq.includes(k));
    const ffnKinds = unique(ffnSeq.filter(f => f !== 'none'));
    const moe = spec.ffn.moe && ffnKinds.includes('moe') ? spec.ffn.moe : null;
    const hasFfn = ffnKinds.length > 0;
    const hasAttn = mixKinds.some(k => AttnKinds.has(k));
    const norm = spec.norm.kind || 'RMSNorm';
    const encoder = spec.family === 'encoder';
    const hasFinal = !(encoder && spec.norm.placement === 'post');
    const learned = spec.position.kind === 'learned';
    const countOf = (seq, k) => seq.filter(v => v === k).length;

    const mixer = { slot: 'mixer', w: mixKinds.length > 1 ? SLOT_W : MIX_W };
    mixer.variants = (mixKinds.length ? mixKinds : ['full']).map(kind => ({
      kind, lines: kindLines(kind, spec, mixKinds.length <= 1), count: mixKinds.length > 1 ? countOf(mixSeq, kind) : null,
    }));
    mixer.h = Math.max(62, 26 + 19 * Math.max(...mixer.variants.map(v => v.lines.length)));
    layoutVariants(mixer);

    // MoE on the right: its leader leaves sideways to the MoE inset; the FFN module leader leaves from a top corner.
    const ffnOrder = ['dense', 'moe'].filter(k => ffnKinds.includes(k));
    const ffn = { slot: 'ffn', w: ffnOrder.length > 1 ? SLOT_W - 40 : 200, h: BOX_H };
    ffn.variants = (ffnOrder.length ? ffnOrder : ['dense']).map(kind => ({
      kind,
      lines: [ffnOrder.length > 1 ? (kind === 'moe' ? 'MoE' : 'Dense FFN') : (kind === 'moe' ? 'MoE' : 'Feed forward')],
      count: ffnOrder.length > 1 ? countOf(ffnSeq, kind) : null,
    }));
    layoutVariants(ffn);

    const nb = (label, variant = 'plain') => ({ lines: [label], variant, w: BOX_W, h: BOX_H });
    const A = () => ({ add: true, h: PLUS_R * 2 });
    const S = () => ({ split: true, h: 0 });
    let flow;
    if (spec.norm.placement === 'sandwich') {
      flow = [S(), nb(`Pre-${norm} 1`), mixer, nb(`Post-${norm} 1`, 'accent'), A(),
        S(), nb(`Pre-${norm} 2`), ffn, nb(`Post-${norm} 2`, 'accent'), A()];
    } else if (spec.norm.placement === 'post' && encoder) {
      flow = [S(), mixer, A(), nb(`${norm} 1`), S(), ffn, A(), nb(`${norm} 2`)];
    } else if (spec.norm.placement === 'post') {
      flow = [S(), mixer, nb(`${norm} 1`, 'accent'), A(), S(), ffn, nb(`${norm} 2`, 'accent'), A()];
    } else {
      flow = [S(), nb(`${norm} 1`), mixer, A(), S(), nb(`${norm} 2`), ffn, A()];
    }
    if (!hasFfn) flow = flow.slice(0, flow.findIndex(f => f.add) + 1);

    const TOP = 34;
    const outBox = at(TOP + 58, BOX_H);
    const finBox = hasFinal ? at(outBox.y1 + GAP, BOX_H) : null;
    const blockTop = (finBox || outBox).y1 + 30;
    const items = flow.slice().reverse();
    let y = blockTop + 24;
    let first = true;
    for (const it of items) {
      if (it.split) {
        it.y = y + GAP / 2 + 2;
        continue;
      }
      // Stacked MoE outlines and count badges need room above the box.
      const lift = it.variants && (it.variants.some(v => v.count) || it.variants.some(v => v.kind === 'moe')) ? 10 : 0;
      if (!first) y += GAP + lift;
      else y += lift;
      first = false;
      Object.assign(it, at(y, it.h));
      y = it.y1;
    }
    const blockBottom = y + GAP + 10;
    let yb = blockBottom + 36;
    let posAdd = null;
    if (learned) {
      posAdd = at(yb, PLUS_R * 2);
      yb = posAdd.y1 + GAP;
    }
    const embBox = at(yb, BOX_H);
    const outerBottom = embBox.y1 + 30;
    const tokBox = at(outerBottom + 34, BOX_H);
    const sampleY = tokBox.y1 + 50;

    const bg = [];
    const lines = [];
    const boxes = [];
    const notes = [];
    bg.push(`<rect x="${OUTER_X0}" y="${outBox.y0 - 28}" width="${OUTER_X1 - OUTER_X0}" height="${outerBottom - outBox.y0 + 28}" rx="30" class="outer"/>`);
    bg.push(`<rect x="${BLOCK_X0}" y="${blockTop}" width="${BLOCK_X1 - BLOCK_X0}" height="${blockBottom - blockTop}" rx="28" class="block"/>`);

    const spine = [tokBox, embBox, ...(posAdd ? [posAdd] : []), ...flow.filter(f => !f.split), ...(finBox ? [finBox] : []), outBox];
    lines.push(path([[CX, sampleY - 20], [CX, tokBox.y1]]));
    for (let i = 0; i + 1 < spine.length; i++) lines.push(...linkUp(spine[i], spine[i + 1]));
    lines.push(path([[CX, outBox.y0], [CX, TOP]]));
    items.forEach((it, i) => {
      if (!it.add) return;
      const split = items.slice(i + 1).find(s => s.split);
      lines.push(path([[CX, split.y], [XRES, split.y], [XRES, it.cy], [CX + PLUS_R, it.cy]]));
      boxes.push(plus(CX, it.cy));
    });
    for (const it of flow) {
      if (it.split || it.add) continue;
      if (!it.variants) {
        boxes.push(box(CX, it.y0, it.w, it.h, it.lines, it.variant));
        continue;
      }
      for (const v of it.variants) {
        boxes.push(it.slot === 'mixer' ? kindBox(v.cx, it.y0, v.w, it.h, v.lines, v.kind) : ffnKindBox(v.cx, it.y0, v.w, it.h, v.lines, v.kind));
        if (v.count) boxes.push(Kit.badge(v.cx + v.w / 2 - 20, it.y0, `${v.count} x`, 'ba'));
      }
    }
    boxes.push(box(CX, outBox.y0, BOX_W, BOX_H, ['Linear output layer']));
    if (finBox) boxes.push(box(CX, finBox.y0, BOX_W, BOX_H, [`Final ${norm}`]));
    boxes.push(box(CX, embBox.y0, 250, BOX_H, ['Token embedding layer']));
    boxes.push(box(CX, tokBox.y0, 190, BOX_H, ['Tokenized text']));
    boxes.push(`<text x="${CX}" y="${sampleY}" class="mono" text-anchor="middle">Sample input text</text>`);

    const mixLeft = CX - mixer.w / 2;
    let ctxAnchor = { x: CX - 125, y: embBox.cy };
    let sideBottom = -1;
    if (learned) {
      const pb = at(posAdd.cy - 25, 50);
      boxes.push(plus(CX, posAdd.cy));
      boxes.push(box(262, pb.y0, 160, 50, ['Positional', 'embedding layer']));
      lines.push(path([[342, posAdd.cy], [CX - PLUS_R, posAdd.cy]]));
      ctxAnchor = { x: 220, y: pb.y1 };
    } else if (hasAttn) {
      const pos = spec.position.kind;
      const label = pos === 'partial RoPE' ? 'Partial RoPE' : pos;
      const ropeY = mixer.cy + (spec.norm.qkNorm ? 16 : 0);
      const rb = at(ropeY - BOX_H / 2, BOX_H);
      boxes.push(box(SIDE_X + SIDE_W / 2, rb.y0, SIDE_W, BOX_H, [label], pos === 'RoPE' ? 'plain' : 'accent'));
      lines.push(path([[SIDE_X + SIDE_W, ropeY], [mixLeft, ropeY]]));
      ctxAnchor = { x: SIDE_X + 34, y: rb.y1 };
      sideBottom = rb.y1;
      if (spec.norm.qkNorm) {
        const qy = mixer.cy - 46;
        boxes.push(box(SIDE_X + SIDE_W / 2, qy - BOX_H / 2, SIDE_W, BOX_H, ['QK-Norm'], 'accent'));
        lines.push(path([[SIDE_X + SIDE_W, qy + 8], [mixLeft, mixer.cy - 14]]));
      }
    }

    const vocab = ['Vocabulary size', `of {${fmtC(spec.vocab)}}`];
    const vocabNote = spec.tiedEmbeddings ? ['output layer shares the', 'token embedding weights'] : [];
    notes.push(text(OUTER_X0 - 16, TOP + 10, vocab, 'co', 'end'));
    if (vocabNote.length) notes.push(text(OUTER_X0 - 16, TOP + 10 + LH * vocab.length, vocabNote, 'note', 'end', 18));
    lines.push(path([[CX - BOX_W / 2, outBox.cy], [OUTER_X0 - 8, TOP + 4 + LH * (vocab.length - 1)]], 'ld', false));

    let leftBottom = TOP + LH * vocab.length + 18 * vocabNote.length;
    if (spec.context) {
      const ctx = wrap(`Supported context length of {${fmtT(spec.context)}} tokens`, LEFT_CHARS);
      const sc = spec.position.scaling;
      if (sc && sc.type && sc.type !== 'default') {
        const name = { yarn: 'YaRN', llama3: 'Llama 3', longrope: 'LongRoPE', dynamic: 'dynamic NTK', linear: 'linear' }[sc.type] || sc.type;
        let tail = `via ${name} scaling`;
        if (sc.factor) tail += ` x${sc.factor}`;
        if (sc.originalContext) tail += ` from {${fmtT(sc.originalContext)}}`;
        ctx.push(...wrap(tail, LEFT_CHARS));
      }
      const nope = types.filter(t => t.rope === false && AttnKinds.has(t.mixer)).length;
      if (nope && !learned) ctx.push(...wrap(`{NoPE} in ${nope} of ${types.length} layers`, LEFT_CHARS));
      const ty = Math.max(ctxAnchor.y + 70, leftBottom + 40);
      lines.push(path([[ctxAnchor.x, ctxAnchor.y + 2], [70, ty - 24]], 'ld', false));
      notes.push(text(12, ty, ctx, 'co'));
      leftBottom = ty + LH * ctx.length;
    }

    const dn = denseNote(spec, ffnSeq);
    if (dn && moe) {
      const dl = wrap(dn, LEFT_CHARS);
      const ty = ffn.cy - 6;
      lines.push(path([[CX - ffn.w / 2, ffn.cy], [OUTER_X0 - 6, ty - 6]], 'ld', false));
      notes.push(text(12, ty, dl, 'co'));
    }

    const repY = sideBottom > blockBottom - 64 ? blockTop + 44 : blockBottom - 24;
    notes.push(`<text x="${BLOCK_X0 - 12}" y="${repY}" class="rep" text-anchor="end">${spec.layers} x</text>`);

    // Column A: FFN module + attention. Column B: MoE + sequence mixers (attention moves there if B is empty).
    // Leaders that would cut through another inset are skipped; title and color carry the link instead.
    const right = [];
    const rightLines = [];
    const cross = [];
    const variantOf = (slot, pred) => slot.variants.find(v => pred(v.kind)) || slot.variants[0];
    const leaderFrom = (slot, v) => [v.cx + v.w / 2, slot.cy];
    const colA = [];
    const colB = [];
    if (hasFfn) colA.push({ id: 'ffn' });
    if (hasAttn) colA.push({ id: 'attention' });
    if (moe) colB.push({ id: 'moe' });
    if (spec.linearMixer) for (const kind of mixKinds.filter(k => SeqKinds.has(k))) colB.push({ id: 'mixer', kind });
    if (!colB.length && colA.length > 1) colB.push(colA.pop());
    if (!colA.length) colA.push(...colB.splice(0));

    const attnBox = () => [...mixer.variants].reverse().find(x => AttnKinds.has(x.kind));
    const placed = [];
    function place(item, x0, y0) {
      if (item.id === 'ffn') {
        const inset = ffnInset(x0, y0, RW, spec);
        right.push(frame(x0, y0, RW, inset.y1 - y0, `FeedForward (${ffnName(spec)}) module`), ...inset.svg);
        let y1 = inset.y1;
        const dim = moe ? moe.expertIntermediate : spec.ffn.intermediate;
        if (dim) {
          const bars = widthBars(x0 + 150, y1 + 24, spec, dim, moe);
          right.push(...bars.svg);
          y1 = bars.y1;
        }
        const dense = variantOf(ffn, k => k === 'dense');
        const from = moe ? [dense.cx + dense.w / 2 - 6, ffn.y0] : leaderFrom(ffn, dense);
        return { y0, y1, frameY1: inset.y1, from, to: [x0 - 4, (y0 + inset.y1) / 2] };
      }
      if (item.id === 'moe') {
        const mi = Insets.moe.draw(spec, { x0, y0, w: RW });
        right.push(...mi.svg);
        return { y0, y1: mi.y1, from: leaderFrom(ffn, variantOf(ffn, k => k === 'moe')), to: [mi.anchor.x - 4, mi.anchor.y], cls: 'lda', anchor: mi.anchor };
      }
      if (item.id === 'attention') {
        const ai = Insets.attention.draw(spec, { x0, y0, w: RW, kinds: mixKinds });
        right.push(...ai.svg);
        const v = attnBox();
        return { y0, y1: ai.y1, from: [v.cx + v.w / 2 - 12, mixer.y1 - 3], to: [ai.anchor.x - 4, ai.anchor.y] };
      }
      const xi = Insets.mixer.draw(spec, { x0, y0, w: RW, kind: item.kind });
      right.push(...xi.svg);
      const v = variantOf(mixer, k => k === item.kind);
      return { y0, y1: xi.y1, from: [v.cx + v.w / 2 - 12, mixer.y1 - 3], to: [xi.anchor.x - 4, xi.anchor.y] };
    }
    // Token-mixer insets sit level with the mixer box when the column above leaves room.
    const alignY = item => (item.id === 'attention' || item.id === 'mixer' ? mixer.y0 - 60 : 0);
    let ay = 62;
    for (const item of colA) {
      const p = place(item, RX_A, Math.max(ay, alignY(item)));
      placed.push({ ...p, col: 'A', item });
      ay = p.y1 + 56;
    }
    let by = 62;
    for (const item of colB) {
      const p = place(item, RX_B, Math.max(by, alignY(item)));
      placed.push({ ...p, col: 'B', item });
      by = p.y1 + 56;
    }

    // A column-B leader is dropped when it would run through a column-A inset (title band included).
    const yAt = (a, b, x) => a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]);
    const blocked = (a, b) => placed.filter(p => p.col === 'A').some(p => {
      const ys = [yAt(a, b, RX_A - 10), yAt(a, b, RX_A + RW + 10)];
      return Math.max(...ys) > p.y0 - 40 && Math.min(...ys) < p.y1;
    });
    for (const p of placed) {
      if (p.col === 'B' && blocked(p.from, p.to)) continue;
      cross.push({ a: p.from, b: p.to, cls: p.cls || 'ld' });
    }
    const ffnPlaced = placed.find(p => p.item.id === 'ffn');
    const moePlaced = placed.find(p => p.item.id === 'moe');
    if (ffnPlaced && moePlaced && ffnPlaced.col !== moePlaced.col) {
      // Every expert is one copy of the FFN module: link the two insets across the gutter.
      rightLines.push(path([[RX_A + RW + 4, (ffnPlaced.y0 + ffnPlaced.frameY1) / 2], [moePlaced.anchor.x - 4, moePlaced.anchor.y]], 'ld', false));
    }

    const extra = [];
    if (spec.attention.softcap) extra.push(`attention logit softcap of {${spec.attention.softcap}}`);
    const mixOnly = mixSeq.filter((m, i) => m !== 'none' && ffnSeq[i] === 'none').length;
    const ffnOnly = mixSeq.filter((m, i) => m === 'none' && ffnSeq[i] !== 'none').length;
    if (ffnOnly) extra.push(`{${mixOnly}} mixer-only and {${ffnOnly}} FFN-only layers`);
    if (extra.length) {
      const ls = extra.flatMap(s => wrap(s, RIGHT_CHARS));
      const ex = by <= ay ? RX_B : RX_A;
      const ey0 = Math.min(ay, by);
      right.push(text(ex, ey0, ls, 'co'));
      if (ex === RX_A) ay = ey0 + LH * ls.length + 56;
      else by = ey0 + LH * ls.length + 56;
    }

    const emb = ['Embedding dimension', `of {${fmtD(spec.hidden)}}`];
    const ey = Math.max(leftBottom + 44, embBox.cy + 18);
    lines.push(path([[CX - 125, embBox.cy], [OUTER_X0 - 8, ey - 6]], 'ld', false));
    notes.push(text(OUTER_X0 - 16, ey, emb, 'co', 'end'));
    leftBottom = ey + LH * emb.length;

    // Same width and scale for every model; the height follows the content so no blank band is reserved.
    const H = Math.ceil(Math.max(sampleY + 24, leftBottom + 16, ay - 40, by - 40));
    const css = Kit.Style;
    const leaders = cross.map(c => path([c.a, c.b], c.cls, false));
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img">` +
      `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto">` +
      `<path d="M0 0 L10 5 L0 10 z" fill="#1a1a1a"/></marker><style>${css}</style></defs>` +
      `<rect width="${W}" height="${H}" fill="#ffffff"/>` +
      bg.join('') + leaders.join('') + rightLines.join('') +
      lines.join('') + boxes.join('') + notes.join('') + right.join('') + '</svg>';
  }

  // one column per layer; the RoPE row only shows up when some layers skip it
  function layerMap(spec) {
    const types = spec.layerTypes || [];
    const n = types.length;
    if (!n) return '';
    const w = 760, lx = 96, cw = (w - lx - 6) / n, rowH = 22, gap = cw > 5 ? 1.5 : 0;
    const hasNope = types.some(t => t.rope === false && AttnKinds.has(t.mixer));
    const rows = [
      ['Token mixer', t => MixerColors[t.mixer] || '#bdbdbd'],
      ['FFN', t => FfnColors[t.ffn] || '#bdbdbd'],
    ];
    if (hasNope) rows.push(['RoPE', t => (AttnKinds.has(t.mixer) ? (t.rope ? '#2b2b2b' : '#ffffff') : '#ececec')]);
    const out = [];
    rows.forEach(([label, color], r) => {
      const y0 = 4 + r * (rowH + 6);
      out.push(`<text x="${lx - 10}" y="${y0 + 15}" class="rowl" text-anchor="end">${label}</text>`);
      types.forEach((t, i) => {
        const fill = color(t);
        const cls = fill === '#ffffff' ? ' class="cell-w"' : '';
        out.push(`<rect x="${(lx + i * cw).toFixed(2)}" y="${y0}" width="${Math.max(0.6, cw - gap).toFixed(2)}" height="${rowH}" fill="${fill}"${cls}><title>layer ${i}: ${esc(MixerNames[t.mixer] || t.mixer)}, ${esc(FfnNames[t.ffn] || t.ffn)}${t.rope === false ? ', NoPE' : ''}</title></rect>`);
      });
    });
    const axisY = 4 + rows.length * (rowH + 6) + 12;
    const step = [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 50, 64].find(s => n / s <= 12) || 100;
    for (let i = 0; i < n; i += step) {
      out.push(`<text x="${(lx + (i + 0.5) * cw).toFixed(1)}" y="${axisY}" class="axis" text-anchor="middle">${i}</text>`);
    }
    out.push(`<text x="${lx + (w - lx) / 2}" y="${axisY + 20}" class="axis" text-anchor="middle">Layer index</text>`);
    const h = axisY + 26;
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="100%" role="img"><style>${Kit.Style}</style>${out.join('')}</svg>`;
  }

  const api = { render, layerMap, ffnName, mixerName, MixerNames, FfnNames, MixerColors, FfnColors };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Diagram = api;
})(this);
