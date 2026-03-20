# <img src="https://github.com/ponhvoan/TruthAnchor/raw/main/ASSETS/anchor.png" width="3%" align="center"></img> TruthAnchor

TruthAnChor (TAC) calibrates raw uncertainty scores of LLM responses into truth-aligned scores. Given score-label pairs $\{S_i, C_i\}_{i=1}^{n}$, where $S\in\{0,1\}$, with $S=1$ indicating a correct response, we approximate the target $p^\star(S)=P(C=1\mid S)$ by learning the mapping:

$$
m_\theta: \mathbb{R} \to [0,1],
$$

where $m_\theta$ is instantiated with a lightweight MLP.

<div align=center>
<img src="https://github.com/ponhvoan/TruthAnchor/raw/main/ASSETS/reliability.png" width="100%" align="center"></img>
</div>

## Usage

### Installation

```bash
conda create -n anchor python=3.11
conda activate anchor
pip install truthanchor
```

### Quick Start

Run the end-to-end example pipeline to reproduce results in the paper. Datasets and models can be modified directly in the Python script.

```bash
python3 examples/tac_eval.py
```

This runs:

1. generation
2. uncertainty scoring
3. mapper training
4. held-out evaluation

The pipeline writes results under ```outputs/<dataset>/<model>/```, including:

- `uncertainty_scores.npz`
- `mapper_eval/results.csv`
- `mapper_eval/scores/*.npz`
- optional comparison and calibration plots

A step-by-step notebook example is available at:```examples/tac_eval_walkthrough.ipynb```. TruthAnchor currently supports:

- response generation for benchmark datasets: TriviaQA, SciQ, and PopQA
- select raw uncertainty score computation
- truth-anchored score mapping with a lightweight MLP
- optional CUE comparison
- metric reporting with AUROC, ECE, and PRR
- plotting for calibration and score comparison
- custom score-label CSV input: first column for uncertainty scores, second for labels with `1 = correct` and `0 = incorrect`.