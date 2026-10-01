// Pure config.json -> architecture spec. Classic script (no ES module) so the page works from file://
// and node tests can require() it.
(function (root) {
  /**
   * @typedef {Object} LayerType
   * @property {'full'|'sliding'|'chunked'|'linear'|'mamba'|'conv'|'none'} mixer
   * @property {'dense'|'moe'|'none'} ffn
   * @property {boolean} rope  false on NoPE layers
   *
   * @typedef {Object} Spec
   * @property {string} modelType      model_type of the language model (text_config's when nested)
   * @property {string} architecture   architectures[0] or ''
   * @property {'decoder'|'encoder'|'encoder-decoder'|'unknown'} family
   * @property {boolean} multimodal    vision/audio tower present (its params are not counted)
   * @property {number} hidden
   * @property {number} layers
   * @property {number} vocab
   * @property {number|null} context
   * @property {boolean} tiedEmbeddings
   * @property {{kind:'RMSNorm'|'LayerNorm', placement:'pre'|'post'|'sandwich', qkNorm:boolean}} norm
   * @property {{kind:'MHA'|'GQA'|'MQA'|'MLA', heads:number, kvHeads:number, headDim:number,
   *   mla:null|{qLoraRank:number|null, kvLoraRank:number, qkNopeHeadDim:number, qkRopeHeadDim:number, vHeadDim:number},
   *   bias:boolean, slidingWindow:number|null, chunkSize:number|null, sinks:boolean, softcap:number|null}} attention
   * @property {{kind:'RoPE'|'partial RoPE'|'learned'|'ALiBi'|'NoPE', theta:number|null, partial:number|null,
   *   scaling:null|{type:string, factor:number|null, originalContext:number|null}}} position
   * @property {{activation:string, gated:boolean, intermediate:number|null,
   *   moe:null|{experts:number, topK:number, sharedExperts:number, expertIntermediate:number,
   *     sharedIntermediate:number|null, router:'softmax'|'sigmoid'}}} ffn
   * @property {null|{kind:string, heads:number|null, headDim:number|null, stateSize:number|null, convKernel:number|null}} linearMixer
   * @property {LayerType[]} layerTypes  length === layers
   * @property {{total:number, active:number, embedding:number, lmHead:number, attention:number,
   *   linearMixer:number, ffnDense:number, ffnExperts:number, ffnShared:number, router:number, norms:number}} params
   * @property {number|null} kvCacheBytesPerToken  bf16, all layers that keep a KV cache (sliding layers counted at full size)
   * @property {string[]} notes     facts worth showing that have no slot above (e.g. 'final logit softcap 30')
   * @property {string[]} warnings  fallbacks taken, unknown fields
   */

  // first key present (and not null/undefined) wins; keys can dot into nested objects
  function pick(obj, keys, fallback) {
    for (const key of keys) {
      const parts = key.split('.');
      let value = obj;
      for (const part of parts) {
        value = value && typeof value === 'object' ? value[part] : undefined;
      }
      if (value !== undefined && value !== null) return value;
    }
    return fallback;
  }

  function findNestedLmConfig(config) {
    for (const key of ['text_config', 'language_config', 'llm_config']) {
      if (config[key] && typeof config[key] === 'object') return config[key];
    }
    return null;
  }

  function hasVisionOrAudio(config) {
    return !!(config.vision_config || config.audio_config);
  }

  // gpt2-family default is 4x hidden when intermediate is not given
  function intermediateFallback(cfg, hidden) {
    const explicit = pick(cfg, ['intermediate_size', 'n_inner', 'ffn_dim', 'd_ff'], null);
    if (explicit !== null) return explicit;
    return 4 * hidden;
  }

  const ENCODER_TYPES = ['bert', 'roberta', 'electra', 'distilbert', 'albert', 'deberta', 'deberta-v2'];
  const ENCODER_DECODER_TYPES = ['t5', 'mt5', 'bart', 'pegasus', 'longt5'];

  function classifyFamily(modelType, architecture) {
    const arch = (architecture || '').toLowerCase();
    if (ENCODER_DECODER_TYPES.indexOf(modelType) !== -1 || /conditionalgeneration.*t5|t5.*conditionalgeneration/.test(arch)) {
      return 'encoder-decoder';
    }
    if (ENCODER_TYPES.indexOf(modelType) !== -1 || /formaskedlm|forsequenceclassification.*bert/.test(arch)) {
      return 'encoder';
    }
    if (/forcausallm|forconditionalgeneration/.test(arch) || modelType) return 'decoder';
    return 'unknown';
  }

  // qk-norm is architectural (not config-gated) for these families
  function qwenThreeStyle(modelType) {
    return modelType === 'qwen3' || modelType === 'qwen3_moe' || modelType === 'qwen3_next' || modelType === 'qwen3_5';
  }

  function detectQkNorm(cfg, modelType) {
    if (qwenThreeStyle(modelType)) return true;
    if (modelType === 'olmo2' || modelType === 'olmo3') return true;
    if (modelType === 'gemma3' || modelType === 'gemma3_text' || modelType === 'gemma3n') return true;
    return !!pick(cfg, ['qk_norm', 'use_qk_norm'], false);
  }

  function detectNorm(cfg, modelType) {
    const layerNormTypes = ['gpt2', 'gpt_neox', 'bert', 'opt', 'phi', 'phi3-v1', 'falcon', 'bloom'];
    const kind = layerNormTypes.indexOf(modelType) !== -1 ? 'LayerNorm' : 'RMSNorm';
    let placement = 'pre';
    if (modelType === 'bert' || modelType === 'olmo2' || modelType === 'olmo3') placement = 'post';
    if (modelType === 'gemma2' || modelType === 'gemma3' || modelType === 'gemma3_text' || modelType === 'gemma3n') {
      placement = 'sandwich';
    }
    return { kind, placement, qkNorm: detectQkNorm(cfg, modelType) };
  }

  function ropeScalingOf(cfg) {
    const scaling = cfg.rope_scaling || (cfg.rope_parameters && cfg.rope_parameters.rope_type ? cfg.rope_parameters : null);
    if (!scaling) return null;
    const type = scaling.rope_type || scaling.type;
    if (!type) return null;
    return {
      type,
      factor: scaling.factor !== undefined ? scaling.factor : null,
      originalContext: scaling.original_max_position_embeddings !== undefined ? scaling.original_max_position_embeddings : null,
    };
  }

  function detectPosition(cfg, modelType, warnings) {
    const posType = cfg.position_embedding_type;
    if (posType === 'absolute' || modelType === 'gpt2' || modelType === 'bert' || modelType === 'opt') {
      return { kind: 'learned', theta: null, partial: null, scaling: null };
    }
    if (modelType === 'bloom' || posType === 'alibi') {
      return { kind: 'ALiBi', theta: null, partial: null, scaling: null };
    }
    if (posType === 'nope') {
      return { kind: 'NoPE', theta: null, partial: null, scaling: null };
    }
    if (modelType === 't5' || modelType === 'mt5') {
      warnings.push('T5 uses a relative-attention-bias position scheme, not absolute learned positions; bucketed as learned');
      return { kind: 'learned', theta: null, partial: null, scaling: null };
    }
    if (modelType === 'mamba' || modelType === 'mamba2') {
      return { kind: 'NoPE', theta: null, partial: null, scaling: null };
    }
    const theta = pick(cfg, ['rope_theta', 'rope_parameters.rope_theta'], null);
    const partialFactor = pick(cfg, ['partial_rotary_factor', 'rotary_pct'], 1);
    const scaling = ropeScalingOf(cfg);
    if (partialFactor < 1) {
      return { kind: 'partial RoPE', theta, partial: partialFactor, scaling };
    }
    if (theta === null) {
      warnings.push('no rope_theta and no learned/ALiBi marker found; defaulting position to RoPE theta 10000');
      return { kind: 'RoPE', theta: 10000, partial: null, scaling };
    }
    return { kind: 'RoPE', theta, partial: null, scaling };
  }

  // bias applies to q/k/v; o_proj bias is a separate, rarer case tracked internally for param math
  function detectBias(cfg, modelType) {
    const qkv = pick(cfg, ['attention_bias', 'qkv_bias'], null);
    if (qkv !== null) return !!qkv;
    if (modelType === 'qwen2' || modelType === 'gpt2' || modelType === 'bert') return true;
    return false;
  }

  function outProjHasBias(modelType, qkvBias) {
    return (modelType === 'gpt2' || modelType === 'bert') || (modelType === 'gpt_oss' && qkvBias);
  }

  function detectAttentionKind(kvHeads, heads, mla) {
    if (mla) return 'MLA';
    if (kvHeads === 1 && heads > 1) return 'MQA';
    if (kvHeads < heads) return 'GQA';
    return 'MHA';
  }

  function detectMla(cfg) {
    if (cfg.kv_lora_rank === undefined || cfg.kv_lora_rank === null) return null;
    return {
      qLoraRank: cfg.q_lora_rank !== undefined ? cfg.q_lora_rank : null,
      kvLoraRank: cfg.kv_lora_rank,
      qkNopeHeadDim: pick(cfg, ['qk_nope_head_dim'], 0),
      qkRopeHeadDim: pick(cfg, ['qk_rope_head_dim'], 0),
      vHeadDim: pick(cfg, ['v_head_dim'], 0),
    };
  }

  function detectAttention(cfg, modelType, hidden, layers, warnings) {
    const mla = detectMla(cfg);
    const heads = pick(cfg, ['num_attention_heads', 'n_head', 'num_heads'], 0);
    let kvHeads;
    if (mla) {
      kvHeads = heads;
    } else if (cfg.multi_query === true) {
      kvHeads = 1;
    } else {
      kvHeads = pick(cfg, ['num_key_value_heads', 'num_kv_heads'], heads);
    }
    let headDim = pick(cfg, ['head_dim'], null);
    if (headDim === null) headDim = heads > 0 ? Math.round(hidden / heads) : 0;
    const qkvBias = detectBias(cfg, modelType);
    const slidingWindow = pick(cfg, ['sliding_window'], null);
    const useSlidingWindow = cfg.use_sliding_window;
    const effectiveSlidingWindow = useSlidingWindow === false ? null : slidingWindow;
    if (heads === 0) {
      warnings.push('no attention heads found (likely a pure state-space model); attention fields are placeholders');
    }
    return {
      kind: detectAttentionKind(kvHeads, heads, mla),
      heads,
      kvHeads,
      headDim,
      mla,
      bias: qkvBias,
      slidingWindow: effectiveSlidingWindow,
      chunkSize: pick(cfg, ['attention_chunk_size'], null),
      sinks: modelType === 'gpt_oss',
      softcap: pick(cfg, ['attn_logit_softcapping'], null),
    };
  }

  function detectMoe(cfg, hidden, intermediateFallbackValue) {
    const experts = pick(cfg, ['num_experts', 'num_local_experts', 'n_routed_experts', 'moe_num_experts'], 0);
    if (!experts) return null;
    const topK = pick(cfg, ['num_experts_per_tok', 'num_experts_per_token', 'moe_topk', 'top_k', 'experts_per_token'], 1);
    let sharedExperts = pick(cfg, ['n_shared_experts', 'num_shared_experts'], null);
    if (sharedExperts === null) {
      const hasShared = cfg.shared_expert_intermediate_size !== undefined ||
        cfg.shared_intermediate_size !== undefined || cfg.intermediate_size_mlp !== undefined;
      sharedExperts = hasShared ? 1 : 0;
    }
    const expertIntermediate = pick(cfg, ['moe_intermediate_size', 'expert_intermediate_size'], intermediateFallbackValue);
    // intermediate_size_mlp (llama4) only signals a shared expert exists; its shared FFN reuses
    // the routed expert width, not that field's value, so it is not a size source here
    const sharedIntermediate = pick(cfg, ['shared_expert_intermediate_size', 'shared_intermediate_size'], null);
    const sigmoidRouter = cfg.scoring_func === 'sigmoid' || cfg.topk_method === 'noaux_tc';
    return {
      experts,
      topK,
      sharedExperts,
      expertIntermediate,
      sharedIntermediate,
      router: sigmoidRouter ? 'sigmoid' : 'softmax',
    };
  }

  function detectLinearMixer(cfg, modelType) {
    if (cfg.linear_num_value_heads !== undefined || cfg.linear_key_head_dim !== undefined) {
      return {
        kind: 'Gated DeltaNet',
        heads: pick(cfg, ['linear_num_value_heads'], null),
        headDim: pick(cfg, ['linear_value_head_dim', 'linear_key_head_dim'], null),
        stateSize: null,
        convKernel: pick(cfg, ['linear_conv_kernel_dim'], null),
      };
    }
    const hasSsmMarker = cfg.mamba_d_state !== undefined || cfg.ssm_state_size !== undefined ||
      cfg.mamba_num_heads !== undefined || cfg.mamba_n_heads !== undefined || cfg.mamba_d_head !== undefined;
    if (modelType === 'mamba' || modelType === 'mamba2' || hasSsmMarker) {
      // 'mamba' (state-spaces/mamba) is the original single-head SSM; granite/nemotron-h/falcon-h1
      // and anything else exposing a per-head SSM shape (mamba_n_heads/mamba_d_head) is Mamba-2 style
      const isOriginalMamba = modelType === 'mamba';
      return {
        kind: isOriginalMamba ? 'Mamba' : 'Mamba-2',
        heads: pick(cfg, ['mamba_num_heads', 'mamba_n_heads'], null),
        headDim: pick(cfg, ['mamba_d_head'], null),
        stateSize: pick(cfg, ['state_size', 'ssm_state_size', 'mamba_d_state'], null),
        convKernel: pick(cfg, ['conv_kernel', 'mamba_d_conv'], null),
      };
    }
    if (cfg.conv_L_cache !== undefined) {
      return {
        kind: 'Short conv',
        heads: null,
        headDim: null,
        stateSize: null,
        convKernel: cfg.conv_L_cache,
      };
    }
    return null;
  }

  const LAYER_TYPE_ALIASES = {
    full_attention: 'full',
    attention: 'full',
    sliding_attention: 'sliding',
    linear_attention: 'linear',
    chunked_attention: 'chunked',
    conv: 'conv',
    mamba: 'mamba',
  };

  function buildLayerTypes(cfg, modelType, layers, attention, moe, linearMixer, warnings) {
    const denseReplace = pick(cfg, ['first_k_dense_replace'], 0);
    const mlpOnlyLayers = cfg.mlp_only_layers || [];
    const decoderSparseStep = pick(cfg, ['decoder_sparse_step'], 1);

    function ffnFor(index) {
      if (!moe) return moe === null && (modelType === 'mamba') ? 'none' : 'dense';
      if (mlpOnlyLayers.indexOf(index) !== -1) return 'dense';
      if (index < denseReplace) return 'dense';
      if (decoderSparseStep > 1 && (index + 1) % decoderSparseStep !== 0) return 'dense';
      return 'moe';
    }

    let result;
    if (Array.isArray(cfg.layer_types) && cfg.layer_types.length === layers) {
      result = cfg.layer_types.map((raw, i) => ({
        mixer: LAYER_TYPE_ALIASES[raw] || 'full',
        ffn: ffnFor(i),
        rope: true,
      }));
    } else if (typeof cfg.hybrid_override_pattern === 'string' && cfg.hybrid_override_pattern.length === layers) {
      result = cfg.hybrid_override_pattern.split('').map((ch) => {
        if (ch === 'M') return { mixer: 'mamba', ffn: 'none', rope: false };
        if (ch === '*') return { mixer: 'full', ffn: 'dense', rope: true };
        if (ch === '-') return { mixer: 'full', ffn: 'none', rope: true };
        if (ch === 'E') return { mixer: 'full', ffn: 'moe', rope: true };
        return { mixer: 'full', ffn: 'dense', rope: true };
      });
    } else if (modelType === 'gemma2') {
      result = Array.from({ length: layers }, (_, i) => ({
        mixer: i % 2 === 0 ? 'sliding' : 'full',
        ffn: 'dense',
        rope: true,
      }));
    } else if (Array.isArray(cfg.full_attn_idxs)) {
      const fullSet = new Set(cfg.full_attn_idxs);
      result = Array.from({ length: layers }, (_, i) => ({
        mixer: fullSet.has(i) ? 'full' : 'conv',
        ffn: 'dense',
        rope: fullSet.has(i),
      }));
    } else if (cfg.sliding_window_pattern) {
      const n = cfg.sliding_window_pattern;
      result = Array.from({ length: layers }, (_, i) => ({
        mixer: (i + 1) % n === 0 ? 'full' : 'sliding',
        ffn: ffnFor(i),
        rope: true,
      }));
    } else if (cfg.full_attention_interval) {
      const n = cfg.full_attention_interval;
      result = Array.from({ length: layers }, (_, i) => ({
        mixer: (i + 1) % n === 0 ? 'full' : 'linear',
        ffn: ffnFor(i),
        rope: (i + 1) % n === 0,
      }));
    } else if (Array.isArray(cfg.no_rope_layers) && cfg.no_rope_layers.length === layers) {
      const chunkSize = attention.chunkSize;
      result = cfg.no_rope_layers.map((usesRope, i) => ({
        mixer: chunkSize && usesRope === 1 ? 'chunked' : 'full',
        ffn: ffnFor(i),
        rope: usesRope === 1,
      }));
    } else if (modelType === 'mamba') {
      result = Array.from({ length: layers }, () => ({ mixer: 'mamba', ffn: 'none', rope: false }));
    } else if (attention.slidingWindow) {
      result = Array.from({ length: layers }, (_, i) => ({ mixer: 'sliding', ffn: ffnFor(i), rope: true }));
    } else {
      result = Array.from({ length: layers }, (_, i) => ({ mixer: 'full', ffn: ffnFor(i), rope: true }));
    }

    // no_rope_layers can coexist with an explicit layer_types/mixer source (e.g. SmolLM3); overlay it
    // onto whichever mixer decision was made above so NoPE layers are not lost
    if (Array.isArray(cfg.no_rope_layers) && cfg.no_rope_layers.length === layers) {
      result = result.map((lt, i) => ({ ...lt, rope: cfg.no_rope_layers[i] === 1 }));
    }

    return result;
  }

  function attnParams(hidden, heads, kvHeads, headDim, qkvBias, oBias) {
    const qDim = heads * headDim;
    const kvDim = kvHeads * headDim;
    const q = hidden * qDim + (qkvBias ? qDim : 0);
    const k = hidden * kvDim + (qkvBias ? kvDim : 0);
    const v = hidden * kvDim + (qkvBias ? kvDim : 0);
    const o = qDim * hidden + (oBias ? hidden : 0);
    return q + k + v + o;
  }

  function mlaParams(hidden, heads, mla) {
    const qLora = mla.qLoraRank;
    const kvLora = mla.kvLoraRank;
    const nope = mla.qkNopeHeadDim;
    const rope = mla.qkRopeHeadDim;
    const vDim = mla.vHeadDim;
    let attn = 0;
    let norms = 0;
    if (qLora) {
      attn += hidden * qLora + qLora * heads * (nope + rope);
      norms += qLora;
    } else {
      attn += hidden * heads * (nope + rope);
    }
    attn += hidden * (kvLora + rope);
    attn += kvLora * heads * (nope + vDim);
    norms += kvLora;
    attn += heads * vDim * hidden;
    return { attn, norms };
  }

  function ffnParams(hidden, intermediate, gated, bias) {
    if (gated) return 3 * hidden * intermediate + (bias ? 2 * intermediate + hidden : 0);
    return 2 * hidden * intermediate + (bias ? intermediate + hidden : 0);
  }

  function mambaBlockParams(hidden, cfg) {
    const dInner = pick(cfg, ['d_inner'], 2 * hidden);
    const convKernel = pick(cfg, ['conv_kernel', 'mamba_d_conv'], 4);
    const stateSize = pick(cfg, ['state_size', 'ssm_state_size', 'mamba_d_state'], 16);
    const dtRank = pick(cfg, ['time_step_rank'], Math.ceil(hidden / 16));
    const useBias = !!cfg.use_bias;
    const useConvBias = cfg.use_conv_bias !== false;
    const inProj = hidden * 2 * dInner;
    const conv = dInner * convKernel + (useConvBias ? dInner : 0);
    const xProj = dInner * (dtRank + 2 * stateSize);
    const dtProj = dtRank * dInner + dInner; // dt_proj always carries its own bias
    const aLog = dInner * stateSize;
    const dSkip = dInner;
    const outProj = dInner * hidden + (useBias ? hidden : 0);
    const norm = hidden;
    return inProj + conv + xProj + dtProj + aLog + dSkip + outProj + norm;
  }

  function shortConvParams(hidden, convKernel) {
    // LFM2-style gated short conv: in_proj to (B, C, x), depthwise conv, out_proj
    return hidden * 3 * hidden + hidden * (convKernel || 3) + hidden * hidden;
  }

  function computeParams(cfg, spec, warnings) {
    const hidden = spec.hidden;
    const vocab = spec.vocab;
    const modelType = spec.modelType;
    const normMul = spec.norm.kind === 'LayerNorm' ? 2 : 1; // LayerNorm carries a bias per dim, RMSNorm does not

    let embedding = vocab * hidden;
    // absolute position table (gpt2/bert/opt keep a learned table; RoPE/ALiBi/T5-relative families do not)
    if (spec.position.kind === 'learned' && (modelType === 'gpt2' || modelType === 'bert' || modelType === 'opt')) {
      embedding += (spec.context || 0) * hidden;
    }
    if (modelType === 'bert') {
      embedding += pick(cfg, ['type_vocab_size'], 0) * hidden;
    }
    const lmHead = spec.tiedEmbeddings ? 0 : vocab * hidden;
    const finalNorm = hidden * (spec.norm.placement === 'sandwich' ? 2 : 1) * normMul;

    let attention = 0;
    let linearMixer = 0;
    let ffnDense = 0;
    let ffnExperts = 0;
    let ffnShared = 0;
    let router = 0;
    let norms = finalNorm;
    let active = embedding + lmHead + finalNorm;

    const a = spec.attention;
    const qkvBias = a.bias;
    const oBias = outProjHasBias(spec.modelType, qkvBias);
    const perLayerNorms = (spec.norm.placement === 'sandwich' ? 4 * hidden : 2 * hidden) * normMul;

    for (let i = 0; i < spec.layers; i++) {
      const lt = spec.layerTypes[i];
      let layerAttn = 0;
      if (lt.mixer === 'full' || lt.mixer === 'sliding' || lt.mixer === 'chunked') {
        if (a.mla) {
          const mla = mlaParams(hidden, a.heads, a.mla);
          layerAttn = mla.attn;
          norms += mla.norms;
        } else {
          layerAttn = attnParams(hidden, a.heads, a.kvHeads, a.headDim, qkvBias, oBias);
        }
        attention += layerAttn;
      } else if (lt.mixer === 'linear' && spec.linearMixer) {
        const lm = spec.linearMixer;
        const vHeads = lm.heads || a.heads;
        const kHeads = pick(cfg, ['linear_num_key_heads'], vHeads);
        const kDim = pick(cfg, ['linear_key_head_dim'], lm.headDim || a.headDim);
        const vDim = lm.headDim || a.headDim;
        const inProj = hidden * (kHeads * kDim * 2 + vHeads * vDim * 2);
        const conv = (kHeads * kDim + vHeads * vDim) * (lm.convKernel || 4);
        const outProj = vHeads * vDim * hidden;
        linearMixer += inProj + conv + outProj;
      } else if (lt.mixer === 'mamba') {
        linearMixer += mambaBlockParams(hidden, cfg);
      } else if (lt.mixer === 'conv' && spec.linearMixer) {
        linearMixer += shortConvParams(hidden, spec.linearMixer.convKernel);
      }
      norms += perLayerNorms;

      const gated = spec.ffn.gated;
      const activationBias = false;
      if (lt.ffn === 'dense') {
        const inter = spec.ffn.intermediate || intermediateFallback(cfg, hidden);
        ffnDense += ffnParams(hidden, inter, gated, activationBias || outProjHasBias(spec.modelType, false));
      } else if (lt.ffn === 'moe' && spec.ffn.moe) {
        const moe = spec.ffn.moe;
        ffnExperts += moe.experts * ffnParams(hidden, moe.expertIntermediate, true, false);
        if (moe.sharedExperts) {
          const sharedInter = moe.sharedIntermediate || moe.expertIntermediate;
          ffnShared += moe.sharedExperts * ffnParams(hidden, sharedInter, true, false);
        }
        router += hidden * moe.experts;
      }
    }

    if (spec.family === 'encoder-decoder') {
      // best-effort: decoder self-attention and cross-attention mirror the encoder's attention shape,
      // and the decoder repeats the encoder's FFN width; exact per-stack config is not modeled
      attention *= 3;
      ffnDense *= 2;
      norms += hidden * spec.layers * normMul; // decoder's extra cross-attention norm per layer
      norms += hidden * normMul; // decoder's own final norm
      const buckets = pick(cfg, ['relative_attention_num_buckets'], 32);
      norms += 2 * buckets * a.heads; // encoder + decoder relative-position bias tables
      warnings.push('encoder-decoder param total is approximated: decoder self/cross-attention and FFN are assumed to mirror the encoder shapes');
    }

    const total = embedding + lmHead + attention + linearMixer + ffnDense + ffnExperts + ffnShared + router + norms;

    let activeFfn;
    if (spec.ffn.moe) {
      const moe = spec.ffn.moe;
      const denseLayerCount = spec.layerTypes.filter((lt) => lt.ffn === 'dense').length;
      const moeLayerCount = spec.layerTypes.filter((lt) => lt.ffn === 'moe').length;
      const perMoeLayerActive = moe.topK * ffnParams(hidden, moe.expertIntermediate, true, false) +
        (moe.sharedExperts ? moe.sharedExperts * ffnParams(hidden, moe.sharedIntermediate || moe.expertIntermediate, true, false) : 0) +
        hidden * moe.experts;
      const denseInter = spec.ffn.intermediate || intermediateFallback(cfg, hidden);
      activeFfn = denseLayerCount * ffnParams(hidden, denseInter, gatedOf(spec), false) + moeLayerCount * perMoeLayerActive;
    } else {
      activeFfn = ffnDense;
    }
    active = embedding + lmHead + attention + linearMixer + activeFfn + norms;

    return {
      total,
      active,
      embedding,
      lmHead,
      attention,
      linearMixer,
      ffnDense,
      ffnExperts,
      ffnShared,
      router,
      norms,
    };
  }

  function gatedOf(spec) {
    return spec.ffn.gated;
  }

  function kvCacheBytesPerToken(spec) {
    const a = spec.attention;
    if (a.mla) {
      const perLayer = (a.mla.kvLoraRank + a.mla.qkRopeHeadDim) * 2;
      const cacheLayers = spec.layerTypes.filter((lt) => lt.mixer === 'full' || lt.mixer === 'sliding' || lt.mixer === 'chunked').length;
      return perLayer * cacheLayers;
    }
    if (!a.heads) return null;
    const perLayer = 2 * a.kvHeads * a.headDim * 2;
    const cacheLayers = spec.layerTypes.filter((lt) => lt.mixer === 'full' || lt.mixer === 'sliding' || lt.mixer === 'chunked').length;
    return perLayer * cacheLayers;
  }

  /** @returns {Spec} */
  function normalize(config) {
    if (!config || typeof config !== 'object') throw new Error('config must be an object');

    const warnings = [];
    const notes = [];
    const architecture = pick(config, ['architectures.0'], '') || '';
    const multimodal = hasVisionOrAudio(config);
    const nested = findNestedLmConfig(config);
    const cfg = nested || config;
    const modelType = pick(cfg, ['model_type'], '') || '';

    const hidden = pick(cfg, ['hidden_size', 'n_embd', 'd_model', 'dim'], 0);
    if (!hidden) warnings.push('hidden size not found; defaulting to 0');
    const layers = pick(cfg, ['num_hidden_layers', 'n_layer', 'num_layers', 'n_layers'], 0);
    if (!layers) warnings.push('layer count not found; defaulting to 0');
    const vocab = pick(cfg, ['vocab_size'], 0);
    const context = pick(cfg, ['max_position_embeddings', 'n_positions', 'seq_length', 'max_sequence_length', 'n_ctx'], null);

    // nested configs (multimodal) rarely carry their own tie flag; the outer config usually does
    const tiedEmbeddings = pick(cfg, ['tie_word_embeddings'], pick(config, ['tie_word_embeddings'], true));

    const norm = detectNorm(cfg, modelType);
    const attention = detectAttention(cfg, modelType, hidden, layers, warnings);
    const position = detectPosition(cfg, modelType, warnings);

    const intermediateExplicit = pick(cfg, ['intermediate_size', 'n_inner', 'ffn_dim', 'd_ff'], null);
    const intermediate = intermediateExplicit !== null ? intermediateExplicit : (modelType === 'gpt2' ? 4 * hidden : null);
    let activation = pick(cfg, ['hidden_act', 'hidden_activation', 'activation_function'], 'unknown');
    if (activation === 'unknown' && (modelType === 't5' || modelType === 'mt5')) activation = 'relu';
    if (activation === 'unknown' && modelType === 'lfm2' && cfg.block_use_swiglu) activation = 'silu';
    const nonGatedTypes = ['gpt2', 'bert', 'gpt_neox', 'opt', 'falcon', 'bloom', 't5', 'mt5'];
    const gated = nonGatedTypes.indexOf(modelType) === -1;
    const moe = detectMoe(cfg, hidden, intermediate || 4 * hidden);

    const linearMixer = detectLinearMixer(cfg, modelType);

    const spec = {
      modelType,
      architecture,
      family: classifyFamily(modelType, architecture),
      multimodal,
      hidden,
      layers,
      vocab,
      context,
      tiedEmbeddings: !!tiedEmbeddings,
      norm,
      attention,
      position,
      ffn: {
        activation,
        gated: moe ? true : gated,
        intermediate: moe ? (cfg.mlp_only_layers && cfg.mlp_only_layers.length ? intermediate : (cfg.first_k_dense_replace ? intermediate : null)) : intermediate,
        moe,
      },
      linearMixer,
      layerTypes: [],
      params: null,
      kvCacheBytesPerToken: null,
      notes,
      warnings,
    };

    spec.layerTypes = buildLayerTypes(cfg, modelType, layers, attention, moe, linearMixer, warnings);

    if (pick(cfg, ['final_logit_softcapping'], null) !== null) {
      notes.push('final logit softcap ' + cfg.final_logit_softcapping);
    }
    if (pick(cfg, ['num_nextn_predict_layers'], 0) > 0) {
      notes.push('checkpoint includes ' + cfg.num_nextn_predict_layers + ' multi-token-prediction layer(s) not modeled in params');
      warnings.push('multi-token-prediction (MTP) layer params are excluded from the total; safetensors total will read higher');
    }

    spec.params = computeParams(cfg, spec, warnings);
    spec.kvCacheBytesPerToken = kvCacheBytesPerToken(spec);

    return spec;
  }

  // 8190000000 -> '8.19B', 124439808 -> '124M', 671e9 -> '671B'
  function formatCount(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return '0';
    const abs = Math.abs(n);
    const units = [
      [1e12, 'T'],
      [1e9, 'B'],
      [1e6, 'M'],
      [1e3, 'K'],
    ];
    for (const [scale, suffix] of units) {
      if (abs >= scale) {
        const value = n / scale;
        const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
        return value.toFixed(digits).replace(/\.0+$/, '') + suffix;
      }
    }
    return String(n);
  }

  // token/size counts from 16384 up that are whole multiples of 1024 print as binary K/M
  // ('131072' -> '128K', '40960' -> '40K', '10485760' -> '10M'); smaller or ragged counts keep
  // thousands separators ('4,096', '7,168'), as they read like dimensions rather than lengths
  function formatTokens(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return '0';
    if (n >= 16384 && n % 1024 === 0) {
      if (n % (1024 * 1024) === 0) return n / (1024 * 1024) + 'M';
      return n / 1024 + 'K';
    }
    return n.toLocaleString('en-US');
  }

  const api = { normalize, formatCount, formatTokens };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Arch = api;
})(this);
