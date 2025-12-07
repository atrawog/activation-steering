# activation-steering

## Project Overview

**activation-steering** is a general-purpose Python library for extracting activation vectors and steering the behavior of Large Language Models (LLMs). It specifically introduces **Conditional Activation Steering**, allowing for precise control where steering is applied only when specific input conditions are met (e.g., "only refuse if the prompt is harmful").

The library is designed to work with Hugging Face `transformers` models and builds upon techniques like Representation Engineering.

## Installation & Setup

This project uses **Poetry** for dependency management.

### Prerequisites
*   Python 3.10+
*   PyTorch
*   Transformers

### Install
```bash
# Clone the repository
git clone https://github.com/IBM/activation-steering
cd activation-steering

# Install in editable mode
pip install -e .
```

## Core Architecture

The library revolves around a few key classes found in the `activation_steering` package:

### 1. `MalleableModel` (`malleable_model.py`)
The primary interface for steering. It wraps a Hugging Face model (`AutoModelForCausalLM`).
*   **`steer(...)`**: Applies a single behavior vector, optionally gated by a condition vector.
*   **`multisteer(...)`**: Applies multiple behavior vectors based on complex logical rules involving multiple condition vectors (e.g., "if C1 or C2 then B1").
*   **`find_best_condition_point(...)`**: Grid search utility to find the optimal layer, threshold, and direction for a condition vector.
*   **`respond(...)`**: Generates text with the active steering configuration.

### 2. `SteeringVector` (`steering_vector.py`)
Represents a direction in the model's activation space.
*   **`train(...)`**: Extracts vectors from a `SteeringDataset` using PCA.
*   **Methods**:
    *   `"pca_pairwise"` (Default/Recommended): Computes direction from negative to positive for each pair, centering on their midpoint.
    *   `"pca_center"`: Standard PCA subtracting global mean.
    *   `"pca_diff"`: Simple difference (Positive - Negative).
*   **Persistence**: Can be saved/loaded using the `.svec` JSON format.

### 3. `SteeringDataset` (`steering_dataset.py`)
Prepares data for vector training.
*   Manages contrastive pairs (positive/negative examples).
*   Handles tokenization and formatting.

### 4. `LeashLayer` (`leash_layer.py`)
Internal component injected into the model's layers. It performs the actual runtime check of activation vs. condition vector and applies the steering intervention.

## Usage Guide

### 1. Extracting a Behavior Vector
Behavior vectors define *how* to steer the model (e.g., "refuse to answer").

```python
from activation_steering import SteeringDataset, SteeringVector

# Create dataset with paired suffixes (compliant vs. non-compliant)
dataset = SteeringDataset(
    tokenizer=tokenizer,
    examples=[("Question", "Question")], # Same prompt, different suffixes
    suffixes=[("Sure!", "I cannot...")]
)

# Extract vector
vector = SteeringVector.train(
    model=model,
    tokenizer=tokenizer,
    steering_dataset=dataset,
    method="pca_pairwise",
    accumulate_last_x_tokens="suffix-only"
)
vector.save("refusal_vector")
```

### 2. Extracting a Condition Vector
Condition vectors define *when* to steer (e.g., "is this prompt harmful?").

```python
# Create dataset with paired prompts (harmful vs. harmless)
dataset = SteeringDataset(
    tokenizer=tokenizer,
    examples=[("Harmful prompt", "Harmless prompt")],
    disable_suffixes=True
)

# Extract vector
cond_vector = SteeringVector.train(
    model=model, 
    tokenizer=tokenizer,
    steering_dataset=dataset,
    method="pca_pairwise",
    accumulate_last_x_tokens="all"
)
```

### 3. Finding the Condition Threshold
Use `find_best_condition_point` to determine the best layer and cosine similarity threshold to distinguish between conditions.

```python
malleable = MalleableModel(model, tokenizer)
best_layer, threshold, direction, f1 = malleable.find_best_condition_point(
    positive_strings=harmful_prompts,
    negative_strings=harmless_prompts,
    condition_vector=cond_vector,
    layer_range=(1, 10),
    threshold_range=(0.0, 0.1)
)
```

### 4. Applying Conditional Steering
```python
malleable.steer(
    behavior_vector=refusal_vector,
    behavior_layer_ids=[15, 16, 17, 18], # Layers to apply steering
    behavior_vector_strength=1.5,
    condition_vector=cond_vector,
    condition_layer_ids=[best_layer],    # Layer to check condition
    condition_vector_threshold=threshold,
    condition_comparator_threshold_is=direction
)

response = malleable.respond("Harmful instruction")
```

## Configuration & Logging

Logging is handled via `GlobalConfig` in `activation_steering/config.py`. It uses the `rich` library for formatted output.

```python
from activation_steering.config import GlobalConfig

# Enable/Disable logging globally or per-class
GlobalConfig.set_verbose(True, "MalleableModel")
GlobalConfig.set_verbose(False, "LeashLayer") # Very verbose if enabled
```

## Key Files & Directories
*   `activation_steering/`: Source code package.
*   `docs/`: Documentation (Quickstart, FAQ).
*   `docs/demo-data/`: JSON datasets for tutorials (Alpaca, refusal, etc.).
*   `CLAUDE.md`: Detailed developer guide and technical context.

---

## AI Attribution (Fedora Policy Compliant)

Per [Fedora AI Contribution Policy](https://docs.fedoraproject.org/en-US/council/policy/ai-contribution-policy/), Gemini **MUST** include the `Assisted-by: Gemini` trailer with a **confidence statement** in all commits:

```
<commit message>

Assisted-by: Gemini (fully tested and validated)
```

### Confidence Statements (Required)

All AI-assisted contributions **MUST** include a confidence statement indicating verification level:

| Statement | When to Use | Evidence |
|-----------|-------------|----------|
| `fully tested and validated` | Testing + all standards met | Complete verification |
| `analysed on a live system` | Observed live system behavior | Partial testing, live analysis |
| `syntax check only` | Pre-commit hooks passed, no functional testing | Linting passed |
| `theoretical suggestion` | No validation performed | AVOID - unverified code |

**MANDATORY for Gemini:**

- **ALWAYS** include confidence statement - non-negotiable
- Trailer goes after commit body, separated by blank line
- Required for ALL Gemini-assisted commits (code, docs, configs)
- Only exception: trivial grammar/spelling corrections

**GitHub Issues and PRs:**

When creating issues or PR descriptions, include at the end:

```markdown
---
*Assisted-by: Gemini (fully tested and validated)*
```
