const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const Arch = require('../arch.js');

function loadFixture(slug) {
  return JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', slug + '.json'), 'utf8'));
}

function normalizeFixture(slug) {
  return Arch.normalize(loadFixture(slug).config);
}

function mixerCounts(spec) {
  const counts = {};
  for (const lt of spec.layerTypes) counts[lt.mixer] = (counts[lt.mixer] || 0) + 1;
  return counts;
}

// --- headline fields, one test per fixture --------------------------------

test('gpt2-xl: learned positions, LayerNorm, non-gated GELU MLP, MHA, tied', () => {
  const spec = normalizeFixture('openai-community-gpt2-xl');
  assert.strictEqual(spec.hidden, 1600);
  assert.strictEqual(spec.layers, 48);
  assert.strictEqual(spec.attention.heads, 25);
  assert.strictEqual(spec.attention.kvHeads, 25);
  assert.strictEqual(spec.attention.kind, 'MHA');
  assert.strictEqual(spec.norm.kind, 'LayerNorm');
  assert.strictEqual(spec.position.kind, 'learned');
  assert.strictEqual(spec.ffn.gated, false);
  assert.strictEqual(spec.tiedEmbeddings, true);
  assert.strictEqual(spec.layerTypes.length, 48);
});

test('Llama-3.2-1B: GQA, RoPE with llama3 scaling', () => {
  const spec = normalizeFixture('unsloth-llama-3-2-1b');
  assert.strictEqual(spec.attention.kind, 'GQA');
  assert.strictEqual(spec.attention.heads, 32);
  assert.strictEqual(spec.attention.kvHeads, 8);
  assert.strictEqual(spec.position.kind, 'RoPE');
  assert.strictEqual(spec.position.scaling.type, 'llama3');
  assert.strictEqual(spec.position.scaling.factor, 32);
  assert.strictEqual(spec.position.scaling.originalContext, 8192);
  assert.strictEqual(spec.layerTypes.length, 16);
});

test('Mistral-7B-v0.1: sliding window on every layer', () => {
  const spec = normalizeFixture('mistralai-mistral-7b-v0-1');
  assert.strictEqual(spec.attention.kind, 'GQA');
  assert.strictEqual(spec.attention.slidingWindow, 4096);
  assert.ok(spec.layerTypes.every((lt) => lt.mixer === 'sliding'));
});

test('Qwen2.5-7B: qkv bias, no sliding despite the field', () => {
  const spec = normalizeFixture('qwen-qwen2-5-7b');
  assert.strictEqual(spec.attention.kind, 'GQA');
  assert.strictEqual(spec.attention.bias, true, 'qwen2 has qkv bias even with no attention_bias field');
  assert.ok(spec.layerTypes.every((lt) => lt.mixer === 'full'), 'use_sliding_window: false disables the window');
});

test('Qwen3-8B: QK-norm', () => {
  const spec = normalizeFixture('qwen-qwen3-8b');
  assert.strictEqual(spec.norm.qkNorm, true);
  assert.strictEqual(spec.attention.kind, 'GQA');
  assert.strictEqual(spec.attention.bias, false);
});

test('Qwen3-30B-A3B: MoE, all layers sparse', () => {
  const spec = normalizeFixture('qwen-qwen3-30b-a3b');
  assert.strictEqual(spec.norm.qkNorm, true);
  assert.ok(spec.ffn.moe);
  assert.strictEqual(spec.ffn.moe.experts, 128);
  assert.strictEqual(spec.ffn.moe.topK, 8);
  assert.ok(spec.layerTypes.every((lt) => lt.ffn === 'moe'));
});

test('Qwen3-Next: hybrid Gated DeltaNet + full attention, shared expert', () => {
  const spec = normalizeFixture('qwen-qwen3-next-80b-a3b-instruct');
  assert.ok(spec.linearMixer);
  assert.strictEqual(spec.linearMixer.kind, 'Gated DeltaNet');
  const counts = mixerCounts(spec);
  assert.strictEqual(counts.linear, 36);
  assert.strictEqual(counts.full, 12);
  assert.strictEqual(spec.ffn.moe.sharedExperts, 1);
  assert.strictEqual(spec.position.kind, 'partial RoPE');
  assert.strictEqual(spec.position.partial, 0.25);
});

test('DeepSeek-V3: MLA, first_k_dense_replace, shared experts, sigmoid router', () => {
  const spec = normalizeFixture('deepseek-ai-deepseek-v3');
  assert.strictEqual(spec.attention.kind, 'MLA');
  assert.ok(spec.attention.mla);
  assert.strictEqual(spec.attention.mla.kvLoraRank, 512);
  const counts = { dense: 0, moe: 0 };
  spec.layerTypes.forEach((lt) => counts[lt.ffn]++);
  assert.strictEqual(counts.dense, 3);
  assert.strictEqual(counts.moe, 58);
  assert.strictEqual(spec.ffn.moe.sharedExperts, 1);
  assert.strictEqual(spec.ffn.moe.router, 'sigmoid');
});

test('Kimi K2: deepseek_v3-shaped config, no MTP layer', () => {
  const spec = normalizeFixture('moonshotai-kimi-k2-instruct');
  assert.strictEqual(spec.attention.kind, 'MLA');
  assert.strictEqual(spec.ffn.moe.experts, 384);
  assert.strictEqual(spec.ffn.moe.router, 'sigmoid');
  assert.strictEqual(spec.notes.length, 0, 'num_nextn_predict_layers is 0 for this checkpoint');
});

test('Gemma 2: alternating sliding/global, softcaps, sandwich norm', () => {
  const spec = normalizeFixture('unsloth-gemma-2-9b');
  assert.strictEqual(spec.norm.placement, 'sandwich');
  assert.strictEqual(spec.attention.softcap, 50);
  assert.ok(spec.notes.some((n) => n.includes('final logit softcap 30')));
  const pattern = spec.layerTypes.map((lt) => lt.mixer).join(',');
  assert.strictEqual(pattern, Array.from({ length: 42 }, (_, i) => (i % 2 === 0 ? 'sliding' : 'full')).join(','));
});

test('Gemma 3: nested text_config, 5:1 sliding pattern, QK-norm, sandwich norm', () => {
  const spec = normalizeFixture('unsloth-gemma-3-4b-it');
  assert.strictEqual(spec.multimodal, true);
  assert.strictEqual(spec.modelType, 'gemma3_text');
  assert.strictEqual(spec.norm.placement, 'sandwich');
  assert.strictEqual(spec.norm.qkNorm, true);
  const counts = mixerCounts(spec);
  // sliding_window_pattern: 6 -> every 6th layer (index 5, 11, 17, 23, 29) is full
  assert.strictEqual(counts.full, 5);
  assert.strictEqual(counts.sliding, 29);
  const fullIndices = spec.layerTypes.map((lt, i) => (lt.mixer === 'full' ? i : null)).filter((i) => i !== null);
  assert.deepStrictEqual(fullIndices, [5, 11, 17, 23, 29]);
});

test('OLMo 2: post-norm placement, QK-norm', () => {
  const spec = normalizeFixture('allenai-olmo-2-1124-7b');
  assert.strictEqual(spec.norm.placement, 'post');
  assert.strictEqual(spec.norm.qkNorm, true);
  assert.strictEqual(spec.attention.kind, 'MHA');
});

test('Mixtral 8x7B: all layers MoE, no shared experts', () => {
  const spec = normalizeFixture('mistralai-mixtral-8x7b-v0-1');
  assert.strictEqual(spec.ffn.moe.experts, 8);
  assert.strictEqual(spec.ffn.moe.topK, 2);
  assert.strictEqual(spec.ffn.moe.sharedExperts, 0);
  assert.ok(spec.layerTypes.every((lt) => lt.ffn === 'moe' && lt.mixer === 'full'));
});

test('gpt-oss-20b: alternating sliding/full via layer_types, sinks, MoE', () => {
  const spec = normalizeFixture('openai-gpt-oss-20b');
  assert.strictEqual(spec.attention.sinks, true);
  const pattern = spec.layerTypes.map((lt) => lt.mixer);
  assert.strictEqual(pattern[0], 'sliding');
  assert.strictEqual(pattern[1], 'full');
  assert.ok(spec.ffn.moe);
  assert.strictEqual(spec.ffn.moe.experts, 32);
  assert.strictEqual(spec.position.scaling.type, 'yarn');
});

test('Phi-3-mini: dense gated MLP', () => {
  const spec = normalizeFixture('microsoft-phi-3-mini-4k-instruct');
  assert.strictEqual(spec.attention.kind, 'MHA');
  assert.strictEqual(spec.ffn.gated, true);
  assert.strictEqual(spec.layerTypes.length, 32);
});

test('SmolLM3: NoPE layers every 4th index', () => {
  const spec = normalizeFixture('huggingfacetb-smollm3-3b');
  const ropeFlags = spec.layerTypes.map((lt) => lt.rope);
  for (let i = 0; i < spec.layers; i++) {
    assert.strictEqual(ropeFlags[i], (i + 1) % 4 !== 0, 'layer ' + i);
  }
});

test('Llama 4 Scout: nested text_config, chunked attention, NoPE layers, interleaved MoE', () => {
  const spec = normalizeFixture('unsloth-llama-4-scout-17b-16e-instruct');
  assert.strictEqual(spec.multimodal, true);
  assert.strictEqual(spec.attention.chunkSize, 8192);
  assert.strictEqual(spec.norm.qkNorm, true);
  const counts = mixerCounts(spec);
  assert.strictEqual(counts.chunked, 36);
  assert.strictEqual(counts.full, 12);
  assert.ok(spec.layerTypes.every((lt) => lt.ffn === 'moe'));
  assert.strictEqual(spec.ffn.moe.sharedExperts, 1);
});

test('GLM-4.5-Air: MoE with a first dense layer, partial rotary', () => {
  const spec = normalizeFixture('zai-org-glm-4-5-air');
  assert.strictEqual(spec.position.kind, 'partial RoPE');
  assert.strictEqual(spec.position.partial, 0.5);
  assert.strictEqual(spec.layerTypes[0].ffn, 'dense');
  assert.strictEqual(spec.layerTypes[1].ffn, 'moe');
  assert.strictEqual(spec.ffn.moe.sharedExperts, 1);
  assert.ok(spec.notes.some((n) => n.includes('multi-token-prediction')));
});

test('mamba-130m: pure SSM, no attention, no explicit position encoding', () => {
  const spec = normalizeFixture('state-spaces-mamba-130m-hf');
  assert.ok(spec.linearMixer);
  assert.strictEqual(spec.linearMixer.kind, 'Mamba', 'the original single-head state-spaces/mamba, not Mamba-2');
  assert.ok(spec.layerTypes.every((lt) => lt.mixer === 'mamba' && lt.ffn === 'none'));
  assert.strictEqual(spec.position.kind, 'NoPE');
  assert.strictEqual(spec.attention.heads, 0);
});

test('granite-4.0-h-tiny: mamba/attention hybrid with MoE FFN on every layer', () => {
  const spec = normalizeFixture('ibm-granite-granite-4-0-h-tiny');
  const counts = mixerCounts(spec);
  assert.strictEqual(counts.mamba, 36);
  assert.strictEqual(counts.full, 4);
  assert.ok(spec.layerTypes.every((lt) => lt.ffn === 'moe'));
  assert.strictEqual(spec.linearMixer.kind, 'Mamba-2', 'per-head SSM shape (mamba_n_heads/mamba_d_head), not the original single-head mamba');
  assert.strictEqual(spec.ffn.moe.sharedExperts, 1, 'shared_intermediate_size is set');
  assert.strictEqual(spec.ffn.moe.sharedIntermediate, 1024);
});

test('LFM2-1.2B: short conv + full attention layer_types', () => {
  const spec = normalizeFixture('liquidai-lfm2-1-2b');
  const counts = mixerCounts(spec);
  assert.strictEqual(counts.full, 6);
  assert.strictEqual(counts.conv, 10);
  assert.strictEqual(spec.linearMixer.kind, 'Short conv');
  const fullIndices = spec.layerTypes.map((lt, i) => (lt.mixer === 'full' ? i : null)).filter((i) => i !== null);
  assert.deepStrictEqual(fullIndices, [2, 5, 8, 10, 12, 14]);
});

test('bert-base-uncased: encoder family, post-norm LayerNorm, does not crash', () => {
  const spec = normalizeFixture('google-bert-bert-base-uncased');
  assert.strictEqual(spec.family, 'encoder');
  assert.strictEqual(spec.norm.kind, 'LayerNorm');
  assert.strictEqual(spec.norm.placement, 'post');
  assert.strictEqual(spec.layerTypes.length, 12);
});

test('t5-small: encoder-decoder family, does not crash', () => {
  const spec = normalizeFixture('google-t5-t5-small');
  assert.strictEqual(spec.family, 'encoder-decoder');
  assert.strictEqual(spec.layerTypes.length, 6);
  assert.ok(spec.warnings.length > 0, 'approximation is disclosed');
});

// --- params total vs safetensors, all fixtures ----------------------------

// Diff tolerance is 3% for text-only decoders. Three families are documented exceptions:
//  - gpt2-xl: the published safetensors also serialize the per-layer causal-mask buffer
//    (h.N.attn.bias, [1,1,1024,1024], confirmed by reading the safetensors header directly),
//    which is not a learned parameter; that alone accounts for the gap (diff ~3.1%).
//  - GLM-4.5-Air: the checkpoint bundles num_nextn_predict_layers extra multi-token-prediction
//    layer(s) that this tool intentionally excludes from the total (recorded in spec.notes /
//    spec.warnings), pushing the diff to ~3.3%. DeepSeek-V3 has the same exclusion but its MTP
//    layer is proportionally small enough to still land inside 3% unassisted.
//  - gemma-3-4b-it: multimodal checkpoint; safetensorsTotal includes the vision tower and
//    projector, which normalize() does not model (text-only by design, diff ~9.8%).
//    Llama-4-Scout is multimodal too but its vision tower is small enough to stay inside 3%.
const TOLERANCE_PCT = 3;
const KNOWN_EXCEPTIONS = new Set(['openai-community-gpt2-xl', 'unsloth-gemma-3-4b-it', 'zai-org-glm-4-5-air']);

const allFixtureFiles = fs.readdirSync(path.join(__dirname, 'fixtures'));
for (const file of allFixtureFiles) {
  const slug = file.replace(/\.json$/, '');
  test('params total vs safetensors: ' + slug, () => {
    const fixture = loadFixture(slug);
    const spec = Arch.normalize(fixture.config);
    assert.ok(spec.params.total > 0);
    if (fixture.safetensorsTotal === null || fixture.safetensorsTotal === undefined) return;
    const diffPct = ((spec.params.total - fixture.safetensorsTotal) / fixture.safetensorsTotal) * 100;
    if (KNOWN_EXCEPTIONS.has(slug)) {
      assert.ok(Math.abs(diffPct) < 15, slug + ' diff% ' + diffPct.toFixed(2) + ' (documented exception, still sane)');
      return;
    }
    assert.ok(Math.abs(diffPct) < TOLERANCE_PCT, slug + ' diff% ' + diffPct.toFixed(2));
  });
}

// --- kvCacheBytesPerToken, hand-computed ----------------------------------

test('kvCacheBytesPerToken: Mistral-7B-v0.1 (GQA, bf16)', () => {
  const spec = normalizeFixture('mistralai-mistral-7b-v0-1');
  // 2 (k+v) * 8 kvHeads * 128 headDim * 2 bytes = 4096 bytes/layer, 32 layers (sliding
  // counted at full size) -> 4096 * 32 = 131072
  assert.strictEqual(spec.kvCacheBytesPerToken, 131072);
});

test('kvCacheBytesPerToken: Llama-3.2-1B (GQA, bf16)', () => {
  const spec = normalizeFixture('unsloth-llama-3-2-1b');
  // 2 * 8 * 64 * 2 = 2048 bytes/layer, 16 layers -> 32768
  assert.strictEqual(spec.kvCacheBytesPerToken, 32768);
});

test('kvCacheBytesPerToken: DeepSeek-V3 (MLA, bf16)', () => {
  const spec = normalizeFixture('deepseek-ai-deepseek-v3');
  // (kvLoraRank 512 + qkRopeHeadDim 64) * 2 bytes = 1152 bytes/layer, 61 layers -> 70272
  assert.strictEqual(spec.kvCacheBytesPerToken, 70272);
});

test('kvCacheBytesPerToken: gpt2-xl (MHA, bf16)', () => {
  const spec = normalizeFixture('openai-community-gpt2-xl');
  // 2 * 25 * 64 * 2 = 6400 bytes/layer, 48 layers -> 307200
  assert.strictEqual(spec.kvCacheBytesPerToken, 307200);
});

// --- formatCount / formatTokens --------------------------------------------

test('formatCount', () => {
  assert.strictEqual(Arch.formatCount(8190000000), '8.19B');
  assert.strictEqual(Arch.formatCount(124439808), '124M');
  assert.strictEqual(Arch.formatCount(671e9), '671B');
  assert.strictEqual(Arch.formatCount(151936), '152K');
  assert.strictEqual(Arch.formatCount(0), '0');
});

test('formatTokens', () => {
  assert.strictEqual(Arch.formatTokens(131072), '128K');
  assert.strictEqual(Arch.formatTokens(32768), '32K');
  assert.strictEqual(Arch.formatTokens(1048576), '1M');
  assert.strictEqual(Arch.formatTokens(4096), '4,096');
  assert.strictEqual(Arch.formatTokens(16384), '16K');
  // >= 16384 and an exact multiple of 1024 (not just a power of two) reads as binary K/M too
  assert.strictEqual(Arch.formatTokens(40960), '40K');
  assert.strictEqual(Arch.formatTokens(163840), '160K');
  assert.strictEqual(Arch.formatTokens(10485760), '10M');
  // below 16384 keeps the thousands separator even when it happens to divide 1024 evenly
  assert.strictEqual(Arch.formatTokens(7168), '7,168');
});

// --- garbage input ----------------------------------------------------------

test('normalize does not throw on an empty config', () => {
  const spec = Arch.normalize({});
  assert.strictEqual(spec.hidden, 0);
  assert.ok(spec.warnings.length > 0);
});

test('normalize does not throw on an unknown model_type', () => {
  const spec = Arch.normalize({ model_type: 'weird' });
  assert.strictEqual(spec.modelType, 'weird');
  assert.ok(Array.isArray(spec.layerTypes));
});

test('normalize throws only on a non-object input', () => {
  assert.throws(() => Arch.normalize(null));
  assert.throws(() => Arch.normalize(5));
  assert.throws(() => Arch.normalize('config'));
  assert.doesNotThrow(() => Arch.normalize({}));
});
