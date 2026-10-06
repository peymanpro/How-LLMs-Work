# Changelog

## 0.2.0 — 2026-10-06

### Added

- Token-level accuracy alongside language-model loss and perplexity.
- Explicit target-range validation for language-model objectives.
- Updated optional integration with the current `How-Transformers-Work` backbone API.
- Python 3.12 GitHub Actions quality checks.
- MIT License.
- LLM mechanics pipeline visual.

### Changed

- Clarified the repository boundary between LLM mechanics and the Transformer core.
- Updated cached inference and integration documentation to distinguish the educational attention-centric cached path from a full production Transformer.
- Reconnected the optional Transformer training/inference bridge after the `How-Transformers-Work` repository rename.
- External Transformer integration tests now skip cleanly when the sibling repository is not available.

### Scope

This remains a small educational NumPy implementation. It does not claim GPT-scale training, production serving, distributed execution, GPU kernels, or a production-grade tokenizer.
