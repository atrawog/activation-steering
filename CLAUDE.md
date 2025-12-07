# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an **activation steering library** for language models, released alongside the ICLR 2025 spotlight paper *Programming Refusal with Conditional Activation Steering*. The library provides tools to:

1. Extract steering vectors from contrastive examples
2. Apply steering vectors to modify model behavior unconditionally
3. Apply **conditional** steering based on input characteristics
4. Compose multiple conditions with logical rules (multi-conditioning)

## Installation

```bash
pip install -e .
```

Uses Poetry for dependency management. Requires Python 3.10+, PyTorch, and Transformers.

## Architecture

### Core Classes

**`SteeringVector`** (`steering_vector.py`)
- Extracts direction vectors via PCA on hidden state differences
- Methods: `pca_pairwise` (recommended), `pca_center`, `pca_diff`
- Save/load with `.svec` file format
- Key function: `SteeringVector.train(model, tokenizer, steering_dataset, method="pca_pairwise")`

**`SteeringDataset`** (`steering_dataset.py`)
- Formats contrastive pairs (positive/negative examples) for training
- Supports chat templates and configurable suffixes
- Outputs `ContrastivePair` objects

**`MalleableModel`** (`malleable_model.py`)
- Wraps any HuggingFace model to enable steering
- `steer()` - Apply single behavior/condition vector
- `multisteer()` - Apply multiple conditions with rules like `"if C1 or C2 then B1"`
- `find_best_condition_point()` - Grid search for optimal layer/threshold/direction
- `respond()` / `respond_batch_sequential()` - Generate text with steering active

**`LeashLayer`** (`leash_layer.py`)
- Internal wrapper applied to each transformer layer
- Handles condition checking (cosine similarity with threshold)
- Applies behavior vectors when conditions are met
- Class-level state tracking for multi-layer coordination

### Data Flow

```
ContrastivePairs → SteeringDataset → SteeringVector.train() → SteeringVector
                                                                    ↓
Model + Tokenizer → MalleableModel → steer(behavior_vector, condition_vector, ...) → Modified generation
```

## Key Parameters

### Steering Configuration
- `behavior_layer_ids`: Layers to apply behavior vector (e.g., `[15, 16, 17, 18, 19, 20, 21, 22, 23]`)
- `behavior_vector_strength`: Multiplier for behavior vector (typically 1.0-2.0)
- `condition_layer_ids`: Layers to check condition (usually single layer, e.g., `[7]`)
- `condition_vector_threshold`: Cosine similarity threshold (0.0-1.0 range)
- `condition_comparator_threshold_is`: `"larger"` or `"smaller"` - when to activate

### Vector Extraction
- `method`: `"pca_pairwise"` (recommended), `"pca_center"`, or `"pca_diff"`
- `accumulate_last_x_tokens`: `"suffix-only"`, `"all"`, or integer (e.g., `1`)

## Common Operations

### Extract a Behavior Vector
```python
from activation_steering import SteeringDataset, SteeringVector

dataset = SteeringDataset(
    tokenizer=tokenizer,
    examples=[(positive_text, negative_text) for ...],
    suffixes=list(zip(behavior_suffixes_pos, behavior_suffixes_neg))
)

vector = SteeringVector.train(
    model=model,
    tokenizer=tokenizer,
    steering_dataset=dataset,
    method="pca_pairwise",
    accumulate_last_x_tokens="suffix-only"
)
vector.save("my_vector")
```

### Apply Unconditional Steering
```python
from activation_steering import MalleableModel, SteeringVector

malleable = MalleableModel(model=model, tokenizer=tokenizer)
vector = SteeringVector.load("my_vector")

malleable.steer(
    behavior_vector=vector,
    behavior_layer_ids=[15, 16, 17, 18, 19, 20, 21, 22, 23],
    behavior_vector_strength=1.5
)

response = malleable.respond("Your prompt here")
```

### Apply Conditional Steering
```python
# First find optimal condition point
best_layers, threshold, direction, f1 = malleable.find_best_condition_point(
    positive_strings=harmful_prompts,
    negative_strings=harmless_prompts,
    condition_vector=condition_vector,
    layer_range=(1, 10),
    threshold_range=(0.0, 0.1),
    threshold_step=0.001
)

# Then steer with condition
malleable.steer(
    behavior_vector=refusal_vector,
    behavior_layer_ids=[15, 16, 17, 18, 19, 20, 21, 22, 23],
    behavior_vector_strength=1.5,
    condition_vector=harmful_condition_vector,
    condition_layer_ids=best_layers,
    condition_vector_threshold=threshold,
    condition_comparator_threshold_is=direction
)
```

### Multi-Condition Steering
```python
malleable.multisteer(
    behavior_vectors=[refusal_vector],
    behavior_layer_ids=[[15, 16, 17, 18, 19, 20, 21, 22, 23]],
    behavior_vector_strengths=[1.5],
    condition_vectors=[cond1, cond2, cond3],
    condition_layer_ids=[[6], [7], [7]],
    condition_vector_thresholds=[0.016, 0.048, 0.02],
    condition_comparator_threshold_is=['larger', 'larger', 'smaller'],
    rules=['if C1 or C2 then B1']  # Logical composition
)
```

## Logging Configuration

Control verbosity via `GlobalConfig`:

```python
from activation_steering.config import GlobalConfig

GlobalConfig.set_verbose(False)  # Disable all logging
GlobalConfig.set_verbose(True, "LeashLayer")  # Enable per-layer logging (very verbose)
```

## Demo Data

Sample datasets are in `docs/demo-data/`:
- `alpaca.json` - General questions for behavior extraction
- `behavior_refusal.json` - Compliant vs non-compliant response suffixes
- `condition_harmful.json` - Harmful vs harmless prompt pairs
- `condition_multiple.json` - Multi-category condition examples

## File Format

Steering vectors are saved as JSON with `.svec` extension:
```json
{
  "model_type": "llama",
  "directions": {"0": [...], "1": [...], ...},
  "explained_variances": {"0": 0.45, "1": 0.38, ...}
}
```

## Model Compatibility

Works with models following either architecture pattern:
- `model.model.layers` (Mistral-like, LLaMA, etc.)
- `model.transformer.h` (GPT-2-like)

The `get_model_layer_list()` utility handles both patterns.

---

## AI Attribution (Fedora Policy Compliant)

Per [Fedora AI Contribution Policy](https://docs.fedoraproject.org/en-US/council/policy/ai-contribution-policy/), Claude **MUST** include the `Assisted-by: Claude` trailer with a **confidence statement** in all commits:

```
<commit message>

Assisted-by: Claude (fully tested and validated)
```

### Confidence Statements (Required)

All AI-assisted contributions **MUST** include a confidence statement indicating verification level:

| Statement | When to Use | Evidence |
|-----------|-------------|----------|
| `fully tested and validated` | Testing + all standards met | Complete verification |
| `analysed on a live system` | Observed live system behavior | Partial testing, live analysis |
| `syntax check only` | Pre-commit hooks passed, no functional testing | Linting passed |
| `theoretical suggestion` | No validation performed | AVOID - unverified code |

**MANDATORY for Claude:**

- **ALWAYS** include confidence statement - non-negotiable
- Trailer goes after commit body, separated by blank line
- Required for ALL Claude-assisted commits (code, docs, configs)
- Only exception: trivial grammar/spelling corrections

**GitHub Issues and PRs:**

When creating issues or PR descriptions, include at the end:

```markdown
---
*Assisted-by: Claude (fully tested and validated)*
```
