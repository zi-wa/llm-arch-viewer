# LLM Architecture Viewer

https://zi-wa.github.io/llm-arch-viewer/

Type a Hugging Face model id and get an architecture diagram drawn from its `config.json`.
The layout follows Sebastian Raschka's [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/),
which only covers models he drew by hand. Everything here is generated in the browser.

![DeepSeek-V3](docs/deepseek-v3.png)

Gated repos like Llama and Gemma fall back to a public re-upload with the same name. You can also drop a local
`config.json` onto the page. Parameter counts come from the config, so weights it doesn't describe (MTP layers,
vision towers) are not counted.

Built with some help from Claude, mostly for parsing configs across model families.
