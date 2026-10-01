# LLM Architecture Viewer

https://zi-wa.github.io/llm-arch-viewer/

Type a Hugging Face model id, get an architecture diagram drawn from its `config.json`.

I liked Sebastian Raschka's [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
but it only covers the models he has drawn by hand, so I wanted the same kind of figure for any model on the Hub.
The layout borrows from his figures; the drawings are generated in the browser and none of his images are used.

![DeepSeek-V3](docs/deepseek-v3.png)

## What it shows

- The block stack: norms (pre / post / sandwich), attention type, FFN or MoE, position encoding, layer count
- Models that mix layer kinds (Gemma 3 sliding/global, Qwen3-Next DeltaNet/attention, Granite 4 Mamba-2/attention)
  get one colored box per kind with its layer count, plus a per-layer strip under the figure
- Detail panels that change shape with the config:
  - attention: query heads grouped onto KV heads (MHA / GQA / MQA), the MLA compression path, and a small mask
    for each attention kind (global, sliding window, chunked, attention sinks)
  - MoE: one cell per expert, with the experts a single token uses filled in
  - FFN width against the embedding width
  - Mamba / Gated DeltaNet / short-conv internals for the non-attention mixers
- Parameter counts per component, computed from the config and compared with the safetensors total the Hub reports

You can also drop a local `config.json` onto the page. Everything runs in the browser; the page only talks to the
Hugging Face API.

Gated repos (Llama, Gemma) can't be read without a login, so the page looks for a public re-upload with the same
name (e.g. `unsloth/...`) and uses its config, which is the same file. The source repo is shown under the title.

## Files

No build step - plain HTML, CSS and JS.

- `arch.js` turns a config into a normalized spec. Most of the work is here, because every model family names
  things differently (`n_embd` vs `hidden_size`, `layer_types` vs `sliding_window_pattern` vs
  `hybrid_override_pattern`, nested `text_config` for multimodal models, ...)
- `diagram.js` draws the main figure; `inset_*.js` draw the detail panels; `svgkit.js` holds the shared SVG helpers
- `app.js` handles the Hub requests, the gated-repo fallback and the tables
- `strings.js` has the UI text (English, Korean)

## Known limits

- Parameter counts come from the config only. Weights the config doesn't describe (DeepSeek's multi-token
  prediction layer, vision towers of multimodal models) are left out, so the total can differ from the
  safetensors number by a few percent
- Encoder-decoder models (T5) are drawn as a single stack
- A new architecture with unusual config fields may parse only partly; the page lists what it couldn't read under
  "Parsed config"

## Notes

I built this with a fair amount of help from Claude (Anthropic), mostly for the config parsing across model
families. I checked the figures against the original configs and papers and changed what looked wrong.
