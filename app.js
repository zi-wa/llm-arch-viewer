// Page glue: resolve a Hugging Face model id, fetch its config.json, render the figure and tables.
(function () {
  const Hf = 'https://huggingface.co';
  // Under serve.py the requests go through its /hf proxy, which adds this PC's token for gated repos.
  const HfApi = ['127.0.0.1', 'localhost'].includes(location.hostname) ? '/hf' : Hf;
  const LangKey = 'llm-arch-viewer.lang';
  const $ = id => document.getElementById(id);
  const input = $('repo');
  const statusLine = $('status');
  const result = $('result');
  let loadSeq = 0;
  let suggestTimer = 0;
  let lang = readLang();
  let shown = null;

  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const num = n => (n === null || n === undefined ? '-' : Number(n).toLocaleString('en-US'));
  const configUrl = repo => `${HfApi}/${repo}/resolve/main/config.json`;

  function t(key, ...args) {
    const value = Strings[lang][key];
    return typeof value === 'function' ? value(...args) : value;
  }

  // Storage can be blocked (private windows, file:// in some browsers); the page then just starts in English.
  function readLang() {
    try {
      return localStorage.getItem(LangKey) === 'ko' ? 'ko' : 'en';
    } catch (err) {
      return 'en';
    }
  }

  function setLang(next) {
    lang = next;
    try {
      localStorage.setItem(LangKey, next);
    } catch (err) {
      // not persisted; the choice still applies to this page view
    }
    document.documentElement.lang = next;
    for (const el of document.querySelectorAll('[data-t]')) el.textContent = t(el.dataset.t);
    for (const el of document.querySelectorAll('[data-t-html]')) el.innerHTML = t(el.dataset.tHtml);
    for (const btn of document.querySelectorAll('#lang button')) btn.setAttribute('aria-pressed', String(btn.dataset.lang === next));
    if (shown) show(shown.title, shown.loaded);
    else document.title = t('title');
  }

  function formatBytes(bytes) {
    const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
    let i = 0;
    let v = bytes;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i++;
    }
    return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
  }

  function parseRepo(raw) {
    const s = raw.trim().replace(/^https?:\/\/(www\.)?(hf\.co|huggingface\.co)\//i, '').split(/[?#]/)[0];
    return s.split('/').filter(Boolean).slice(0, 2).join('/');
  }

  async function fetchJson(url) {
    const res = await fetch(url);
    if (!res.ok) throw Object.assign(new Error(`HTTP ${res.status}`), { status: res.status });
    return res.json();
  }

  const searchModels = (q, limit) =>
    fetchJson(`${HfApi}/api/models?search=${encodeURIComponent(q)}&sort=downloads&direction=-1&limit=${limit}`);

  async function resolveRepo(query) {
    if (query.includes('/')) return query;
    const hits = await searchModels(query, 20);
    const exact = hits.find(h => h.id.split('/')[1].toLowerCase() === query.toLowerCase());
    if (!exact && !hits.length) throw new Error(t('noResults', query));
    return (exact || hits[0]).id;
  }

  // Gated repos (Llama, Gemma) answer 401 on files, but public re-uploads under the same name carry the same config.
  async function mirrorConfig(repo) {
    const name = repo.split('/')[1].toLowerCase();
    const hits = await searchModels(name, 40);
    const mirrors = hits.filter(h => h.id !== repo && h.id.split('/')[1].toLowerCase() === name).slice(0, 6);
    for (const m of mirrors) {
      try {
        return { config: await fetchJson(configUrl(m.id)), source: m.id };
      } catch (err) {
        if (![401, 403, 404].includes(err.status)) throw err;
      }
    }
    return null;
  }

  async function loadModel(repo, depth = 0) {
    const [metaRes, cfgRes] = await Promise.allSettled([fetchJson(`${HfApi}/api/models/${repo}`), fetchJson(configUrl(repo))]);
    const meta = metaRes.status === 'fulfilled' ? metaRes.value : null;
    if (cfgRes.status === 'fulfilled') return { config: cfgRes.value, source: repo, meta };
    if (!(cfgRes.reason.status > 0)) throw cfgRes.reason;
    if (!meta) throw new Error(t('repoMissing', repo));
    const code = cfgRes.reason.status;
    if (code === 401 || code === 403) {
      const mirror = await mirrorConfig(repo);
      if (mirror) return { ...mirror, meta };
      throw new Error(t('gatedNoMirror', repo));
    }
    // GGUF-only and adapter repos have no config.json of their own; the card's base_model points at one.
    const base = meta.cardData && meta.cardData.base_model;
    const baseRepo = Array.isArray(base) ? base[0] : base;
    if (code === 404 && baseRepo && depth < 2) return loadModel(baseRepo, depth + 1);
    throw new Error(t('noConfig', repo, code));
  }

  function setStatus(message, isError = false) {
    statusLine.textContent = message;
    statusLine.title = message;
    statusLine.dataset.error = isError ? '1' : '';
  }

  function specRows(spec) {
    const a = spec.attention;
    const fmtT = Arch.formatTokens;
    const types = spec.layerTypes || [];
    const rows = [
      [t('rowArchitecture'), `${spec.architecture || '-'} (${spec.modelType || '-'})`],
      [t('rowFamily'), t('families')[spec.family] || spec.family],
      [t('rowLayers'), num(spec.layers)],
      [t('rowHidden'), num(spec.hidden)],
      [t('rowVocab'), num(spec.vocab)],
      [t('rowContext'), spec.context ? t('tokens', num(spec.context)) : '-'],
    ];
    if (a.heads) {
      let attn = a.kind === 'MLA' && a.mla
        ? `MLA, heads ${a.heads}, query latent ${a.mla.qLoraRank ? num(a.mla.qLoraRank) : '-'}, KV latent ${num(a.mla.kvLoraRank)}, ` +
          `RoPE dim ${a.mla.qkRopeHeadDim}, nope dim ${a.mla.qkNopeHeadDim}, value dim ${a.mla.vHeadDim}`
        : `${a.kind}, query heads ${a.heads}, KV heads ${a.kvHeads}, head dim ${a.headDim}`;
      if (a.bias) attn += ', QKV bias';
      if (a.sinks) attn += ', attention sinks';
      if (a.softcap) attn += `, logit softcap ${a.softcap}`;
      rows.push([t('rowAttention'), attn]);
    }
    if (a.slidingWindow && types.some(x => x.mixer === 'sliding')) rows.push([t('rowSliding'), t('tokens', num(a.slidingWindow))]);
    if (a.chunkSize && types.some(x => x.mixer === 'chunked')) rows.push([t('rowChunk'), t('tokens', num(a.chunkSize))]);
    const p = spec.position;
    let pos = p.kind;
    if (p.theta) pos += `, theta ${num(p.theta)}`;
    if (p.partial) pos += `, ${t('rotaryFraction', p.partial)}`;
    if (p.scaling && p.scaling.type && p.scaling.type !== 'default') {
      pos += `, ${p.scaling.type} scaling`;
      if (p.scaling.factor) pos += ` x${p.scaling.factor}`;
      if (p.scaling.originalContext) pos += ` ${t('scaledFrom', fmtT(p.scaling.originalContext))}`;
    }
    const nope = types.filter(x => x.rope === false).length;
    if (nope && p.kind !== 'learned') pos += `, ${t('nopeLayers', nope)}`;
    rows.push([t('rowPosition'), pos]);
    const placement = t('placements')[spec.norm.placement] || spec.norm.placement;
    rows.push([t('rowNorm'), `${spec.norm.kind}, ${placement}${spec.norm.qkNorm ? ', QK-Norm' : ''}`]);
    if (types.some(x => x.ffn !== 'none')) {
      rows.push([t('rowFfn'), `${Diagram.ffnName(spec)} (${spec.ffn.activation || '-'}), hidden ${num(spec.ffn.intermediate)}`]);
    }
    const m = spec.ffn.moe;
    if (m) {
      rows.push([t('rowMoe'), `experts ${m.experts}, top-${m.topK}, shared ${m.sharedExperts}, expert hidden ${num(m.expertIntermediate)}` +
        `${m.sharedIntermediate ? `, shared hidden ${num(m.sharedIntermediate)}` : ''}, ${m.router} router`]);
    }
    const lm = spec.linearMixer;
    if (lm) {
      const bits = [lm.kind];
      if (lm.heads) bits.push(`heads ${lm.heads}`);
      if (lm.headDim) bits.push(`head dim ${lm.headDim}`);
      if (lm.stateSize) bits.push(`state ${lm.stateSize}`);
      if (lm.convKernel) bits.push(`conv kernel ${lm.convKernel}`);
      rows.push([t('rowMixer'), bits.join(', ')]);
    }
    if (spec.kvCacheBytesPerToken) {
      let kv = t('kvPerToken', formatBytes(spec.kvCacheBytesPerToken));
      if (spec.context) kv += `, ${t('kvAtContext', fmtT(spec.context), formatBytes(spec.kvCacheBytesPerToken * spec.context))}`;
      rows.push([t('rowKv'), kv]);
    }
    rows.push([t('rowTied'), spec.tiedEmbeddings ? t('tiedYes') : t('tiedNo')]);
    if (spec.multimodal) rows.push([t('rowMultimodal'), t('multimodalText')]);
    return rows;
  }

  function paramTable(spec, meta) {
    const P = spec.params;
    const m = spec.ffn.moe;
    const fmtC = Arch.formatCount;
    const mixer = spec.linearMixer ? spec.linearMixer.kind : 'Sequence mixer';
    const rows = [
      ['Token embedding', P.embedding],
      ['Linear output layer', P.lmHead],
      ['Attention', P.attention],
      [mixer, P.linearMixer],
      ['Dense FFN', P.ffnDense],
      ['Routed experts', P.ffnExperts],
      ['Shared experts', P.ffnShared],
      ['Router', P.router],
      ['Norms', P.norms],
    ].filter(r => r[1] > 0);
    const activeOf = (label, v) => (label === 'Routed experts' && m ? (v * m.topK) / m.experts : v);
    const head = `<thead><tr><th>${t('colComponent')}</th><th>${t('colParams')}</th>` +
      `${m ? `<th>${t('colActive')}</th>` : ''}<th>${t('colShare')}</th></tr></thead>`;
    const body = rows.map(([label, v]) => {
      const share = P.total ? (100 * v) / P.total : 0;
      return `<tr><td>${esc(label)}</td><td>${fmtC(v)}</td>${m ? `<td>${fmtC(activeOf(label, v))}</td>` : ''}` +
        `<td><span style="width:${share.toFixed(2)}%"></span><em>${share < 0.1 ? '<0.1' : share.toFixed(1)}%</em></td></tr>`;
    }).join('');
    let foot = `<tr><th>${t('totalComputed')}</th><td>${fmtC(P.total)}</td>${m ? `<td>${fmtC(P.active)}</td>` : ''}<td></td></tr>`;
    const st = meta && meta.safetensors && meta.safetensors.total;
    if (st) {
      const diff = P.total ? ((P.total - st) / st) * 100 : 0;
      foot += `<tr><th>${t('totalSafetensors')}</th><td>${fmtC(st)}</td>${m ? '<td></td>' : ''}` +
        `<td><em>${t('diff', `${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%`)}</em></td></tr>`;
    }
    return head + `<tbody>${body}</tbody><tfoot>${foot}</tfoot>`;
  }

  function legend(spec) {
    const types = spec.layerTypes || [];
    const mix = {};
    const ffn = {};
    for (const x of types) {
      mix[x.mixer] = (mix[x.mixer] || 0) + 1;
      ffn[x.ffn] = (ffn[x.ffn] || 0) + 1;
    }
    const item = (color, label, n) =>
      `<li><span style="background:${color}"${color === '#ffffff' ? ' data-empty="1"' : ''}></span>${esc(label)} <b>${n}</b></li>`;
    const out = Object.keys(mix).map(k => item(Diagram.MixerColors[k] || '#bdbdbd', Diagram.mixerName(k, spec), mix[k]));
    out.push(...Object.keys(ffn).map(k => item(Diagram.FfnColors[k] || '#bdbdbd', `FFN: ${Diagram.FfnNames[k] || k}`, ffn[k])));
    const nope = types.filter(x => x.rope === false && ['full', 'sliding', 'chunked'].includes(x.mixer)).length;
    if (nope) out.push(item('#ffffff', t('legendNope'), nope));
    return out.join('');
  }

  function show(title, loaded) {
    const spec = Arch.normalize(loaded.config);
    const meta = loaded.meta;
    const fmtC = Arch.formatCount;
    shown = { title, loaded };
    $('name').textContent = title;
    const sub = [spec.architecture, spec.modelType].filter(Boolean).join(' / ');
    const via = loaded.local ? t('viaLocal', loaded.source) : loaded.source !== title ? t('viaMirror', loaded.source) : '';
    $('sub').textContent = sub + via;

    const facts = [[t('factParams'), fmtC(spec.params.total)]];
    if (spec.ffn.moe) facts.push([t('factActive'), fmtC(spec.params.active)]);
    facts.push([t('factLayers'), num(spec.layers)]);
    if (spec.context) facts.push([t('factContext'), Arch.formatTokens(spec.context)]);
    if (spec.kvCacheBytesPerToken) facts.push([t('factKv'), formatBytes(spec.kvCacheBytesPerToken)]);
    const license = meta && meta.cardData && meta.cardData.license;
    if (license) facts.push([t('factLicense'), String(license)]);
    if (meta && meta.createdAt) facts.push([t('factCreated'), meta.createdAt.slice(0, 10)]);
    $('facts').innerHTML = facts.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('');

    $('diagram').innerHTML = Diagram.render(spec);
    $('caption').innerHTML = loaded.local
      ? esc(t('captionLocal', loaded.source))
      : t('captionRemote', `<a href="${Hf}/${esc(loaded.source)}/blob/main/config.json">${esc(loaded.source)}</a>`);
    $('layermap').innerHTML = Diagram.layerMap(spec);
    $('legend').innerHTML = legend(spec);
    $('params').innerHTML = paramTable(spec, meta);
    $('spec').innerHTML = specRows(spec).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
    const notes = [...(spec.notes || []), ...(spec.warnings || []).map(w => t('notePrefix') + w)];
    $('notes').innerHTML = notes.map(n => `<li>${esc(n)}</li>`).join('');
    result.hidden = false;
    document.title = t('docTitle', title);
  }

  async function load(query) {
    if (!query) return;
    const seq = ++loadSeq;
    input.value = query;
    setStatus(t('loading', query));
    try {
      const repo = await resolveRepo(query);
      const loaded = await loadModel(repo);
      if (seq !== loadSeq) return;
      if (repo !== query) history.replaceState(null, '', `#${repo}`);
      input.value = repo;
      show(repo, loaded);
      setStatus('');
    } catch (err) {
      if (seq !== loadSeq) return;
      console.error(err);
      result.hidden = true;
      shown = null;
      setStatus(err instanceof TypeError ? t('networkError', err.message) : err.message, true);
    }
  }

  async function openFile(file) {
    if (!file) return;
    const seq = ++loadSeq;
    try {
      const config = JSON.parse(await file.text());
      if (seq !== loadSeq) return;
      // Saved configs often carry the repo they came from; the file name is the fallback title.
      const title = typeof config._name_or_path === 'string' && config._name_or_path ? config._name_or_path : file.name;
      history.replaceState(null, '', location.pathname + location.search);
      input.value = '';
      show(title, { config, source: file.name, meta: null, local: true });
      setStatus('');
    } catch (err) {
      if (seq !== loadSeq) return;
      console.error(err);
      result.hidden = true;
      shown = null;
      setStatus(err instanceof SyntaxError ? t('notJson', file.name) : `${file.name}: ${err.message}`, true);
    }
  }

  for (const btn of document.querySelectorAll('#lang button')) btn.addEventListener('click', () => setLang(btn.dataset.lang));

  $('file').addEventListener('change', e => {
    openFile(e.target.files[0]);
    e.target.value = '';
  });
  document.addEventListener('dragover', e => {
    if (!e.dataTransfer.types.includes('Files')) return;
    e.preventDefault();
    document.body.dataset.drag = '1';
  });
  document.addEventListener('dragleave', e => {
    if (!e.relatedTarget) document.body.dataset.drag = '';
  });
  document.addEventListener('drop', e => {
    if (!e.dataTransfer.files.length) return;
    e.preventDefault();
    document.body.dataset.drag = '';
    openFile(e.dataTransfer.files[0]);
  });

  const hashRepo = () => parseRepo(decodeURIComponent(location.hash.slice(1)));

  $('query').addEventListener('submit', e => {
    e.preventDefault();
    const repo = parseRepo(input.value);
    if (!repo) return;
    if (hashRepo() === repo) load(repo);
    else location.hash = repo;
  });
  window.addEventListener('hashchange', () => load(hashRepo()));

  input.addEventListener('input', () => {
    clearTimeout(suggestTimer);
    const q = input.value.trim();
    if (q.length < 2 || q.includes('://')) return;
    suggestTimer = setTimeout(async () => {
      // Suggestions are optional; a failed search leaves the list as it was.
      const hits = await searchModels(q, 10).catch(() => null);
      if (hits) $('suggest').innerHTML = hits.map(h => `<option value="${esc(h.id)}">`).join('');
    }, 250);
  });

  if (lang !== 'en') setLang(lang);
  if (location.hash.length > 1) load(hashRepo());
})();
