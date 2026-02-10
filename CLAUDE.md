# CLAUDE.md - MaxText Repository Guide

## Project Overview

MaxText is a high-performance, highly scalable, open-source LLM training library written in pure Python/JAX, targeting Google Cloud TPUs and GPUs. It supports pre-training, post-training (SFT, RL, DPO, distillation), multimodal training, and inference for 40+ model architectures (Llama, Gemma, DeepSeek, Qwen, Mistral, GPT, OLMo, and more).

- **License:** Apache 2.0
- **Python:** 3.12+
- **Core frameworks:** JAX, Flax (Linen + NNX), Optax, Orbax
- **Package manager:** Hatchling (modern setuptools)
- **Repository:** https://github.com/AI-Hypercomputer/maxtext

## Repository Structure

```
src/
  MaxText/          # Core training/inference framework (legacy PascalCase package)
    train.py        # Main training entry point
    pyconfig.py     # Configuration management (Pydantic-based)
    sharding.py     # JAX distributed sharding
    maxengine.py    # Inference engine
    layers/         # Model layer implementations (attentions, embeddings, etc.)
      models.py     # Main transformer model
      gemma*.py, llama*.py, deepseek.py, mistral.py, qwen3.py  # Per-model layers
    input_pipeline/ # Data loading (Grain, HuggingFace, TFDS)
    rl/             # Reinforcement learning training
    sft/            # Supervised fine-tuning
  maxtext/          # Modern modular package (lowercase)
    configs/        # YAML configuration files
      base.yml      # Base configuration template
      models/       # Model-specific YAML configs (40+ models)
      tpu/          # TPU-specific configs (v4, v5e, v5p, v6e)
      gpu/          # GPU-specific configs
    trainers/       # Training implementations
      post_train/   # SFT, DPO, GRPO, distillation trainers
    inference/      # Inference/serving (paged attention, KV cache, JetStream)
    multimodal/     # Vision-language model support (Gemma 3, Llama 4)
    checkpoint_conversion/  # HF/MaxText checkpoint converters
    common/         # Shared utilities (checkpointing, data_loader, profiler)
    utils/          # Utility functions (logging, model creation, GCS, LoRA)
tests/
  unit/             # Unit tests
  integration/      # Integration tests (GCS, system-level)
  end_to_end/       # End-to-end tests (tpu/, gpu/)
  inference/        # Inference-specific tests
  assets/           # Test fixtures (golden_logits, local_datasets)
dependencies/
  requirements/     # Dependency files
  dockerfiles/      # Docker images for TPU/GPU
tools/
  dev/              # Development scripts (linting, testing)
  setup/            # Environment setup scripts
  orchestration/    # Multi-host training orchestration
docs/               # Sphinx documentation (hosted on ReadTheDocs)
benchmarks/         # Benchmarking infrastructure
```

**Note on dual package structure:** The repo has both `src/MaxText/` (original) and `src/maxtext/` (modern restructured). Both are actively used. See `RESTRUCTURE.md` for rationale.

## Build and Install

```bash
# Install from source (TPU)
pip install -e ".[tpu]"

# Install from source (GPU/CUDA 12)
pip install -e ".[cuda12]"

# Install from PyPI
pip install maxtext[tpu]
pip install maxtext[cuda12]
```

## Running Tests

```bash
# Run all default tests (respects pytest.ini ignores/markers)
python3 -m pytest tests/

# CPU-only unit tests
python3 -m pytest tests/ -m "cpu_only"

# TPU tests (excludes CPU/GPU)
python3 -m pytest tests/ -m "not cpu_only and not gpu_only"

# GPU tests (excludes CPU/TPU)
python3 -m pytest tests/ -m "not cpu_only and not tpu_only"

# Integration tests
python3 -m pytest tests/ -m "integration_test"

# Run a specific test file
python3 -m pytest tests/unit/some_test.py
```

**Test file naming:** `*_test.py` or `*_tests.py`

**Test markers** (defined in `pytest.ini`):
- `tpu_only` - TPU hardware required
- `gpu_only` - GPU hardware required
- `cpu_only` - CPU tests
- `decoupled` - Offline/GCP-independent mode
- `scheduled_only` - Scheduled CI runs only
- `integration_test` - Integration tests (GCS, system-level)
- `external_serving` - JetStream/serving tests
- `external_training` - Goodput integration tests

**Note:** Many hardware-specific tests are ignored by default in `pytest.ini`. Check that file before wondering why tests aren't discovered.

## Code Quality and Linting

Pre-commit hooks are configured in `.pre-commit-config.yaml`:

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Run specific checks
pre-commit run pylint --all-files
pre-commit run pyink --all-files
pre-commit run codespell --all-files
```

### Tools

| Tool | Version | Purpose | Key Config |
|------|---------|---------|------------|
| **pyink** | 24.10.1 | Python formatter (Google's Black fork) | 2-space indent, 122 char line length |
| **pylint** | 3.3.8 | Linter (Google style) | Disabled: R0401, R0917, W0201, W0613 |
| **codespell** | 2.4.1 | Spell checker | Ignores: ND, nd, sems, TE, ROUGE, etc. |
| **mdformat** | 0.7.22 | Markdown formatter | Only applies to `docs/` |

### Quick lint + test script

```bash
bash tools/dev/unit_test_and_lint.sh
```

## Code Style and Conventions

### Formatting
- **Indentation:** 2 spaces (enforced by pyink)
- **Line length:** 122 characters max
- **Formatter:** pyink (Google's opinionated Python formatter based on Black)
- **Style guide:** Google Python Style Guide

### Naming
- `snake_case` for functions, methods, variables
- `CamelCase` for classes and Enums
- `UPPER_CASE` for module-level constants
- `_prefix` for private functions/methods
- `snake_case` for config field names

### Imports (ordering)
```python
# 1. Copyright header (Apache 2.0) - required on all files
# 2. pylint directives
# 3. Module docstring
# 4. Standard library
# 5. Third-party (JAX, Flax, NumPy, etc.)
# 6. Internal MaxText imports
```

### Docstrings
- Google-style docstrings (Args, Returns, Raises sections)
- Type annotations in function signatures, not in docstrings

### Logging
- Use `maxtext.utils.max_logging` instead of standard `logging`
- Usage: `max_logging.log(f"message {var}")`

### Models
- Models are Flax Linen `nn.Module` subclasses
- Use `setup()` for layer initialization
- Support multiple modes: `MODEL_MODE_TRAIN`, `MODEL_MODE_PREFILL`, `MODEL_MODE_AUTOREGRESSIVE`
- Sharding via JAX logical axis names and explicit sharding constraints

### Configuration
- YAML-based configs in `src/maxtext/configs/` with hierarchical inheritance
- Python-side validation via Pydantic (`BaseModel`) in `pyconfig.py`
- CLI overrides: `python -m MaxText.train src/maxtext/configs/base.yml model_name=llama3.1-70b run_name=my_run`

## CI/CD Pipeline

CI runs on GitHub Actions (`.github/workflows/`):

1. **CodeQuality.yml** - Pre-commit checks (pylint, pyink, codespell) on PRs
2. **build_and_test_maxtext.yml** - Main test pipeline:
   - Doc-only change detection (skips tests for .md/.ipynb-only PRs)
   - CPU unit tests (Python 3.12, 2 parallel worker groups)
   - TPU unit + integration tests (v6e-4)
   - GPU unit + integration tests (A100-40GB x4)
   - Pathways (multi-host) tests
   - Jupyter notebook execution
3. **check_docs_build.yml** - Sphinx documentation build validation
4. **pypi_release.yml** - PyPI package release

## Key Entry Points

| Entry Point | Description |
|-------------|-------------|
| `MaxText.train` | Main pre-training loop |
| `MaxText.train_compile` | AOT compilation |
| `MaxText.sft_trainer` | Supervised fine-tuning |
| `maxtext.trainers.post_train.dpo` | DPO training |
| `maxtext.trainers.post_train.grpo` | GRPO/GSPO RL training |
| `maxtext.inference.maxengine` | Inference engine |
| `maxtext.checkpoint_conversion` | Checkpoint format conversion |

## Dependencies

Core dependencies are in `dependencies/requirements/requirements.txt`. Platform-specific deps:
- **TPU:** `dependencies/requirements/generated_requirements/tpu-requirements.txt`
- **GPU (CUDA 12):** `dependencies/requirements/generated_requirements/cuda12-requirements.txt`
- **Docs:** `dependencies/requirements/requirements_docs.txt`

## Contributing

- Google CLA required
- All submissions require code review via GitHub PRs
- See `CONTRIBUTING.md` for full details
- PR template at `.github/PULL_REQUEST_TEMPLATE.md`

## Common Patterns to Know

- **Gradient accumulation:** `MaxText.gradient_accumulation` module
- **Sequence packing:** `MaxText.sequence_packing` / `maxtext.input_pipeline.packing`
- **Vocabulary tiling:** `MaxText.vocabulary_tiling` for distributed embeddings
- **Checkpoint conversion:** Scripts in `maxtext/checkpoint_conversion/` for HF <-> MaxText
- **LoRA:** `maxtext.utils.lora_utils` for adapter-based fine-tuning
- **Quantization:** `MaxText.layers.quantizations` and `MaxText.layerwise_quantization`
- **Multi-host orchestration:** `tools/orchestration/` for multi-node training

## Documentation

- Sphinx docs at `docs/` (hosted at https://maxtext.readthedocs.io/)
- Build docs: `sphinx-build -M html docs out`
- Doc dependencies: `pip install -r dependencies/requirements/requirements_docs.txt`
