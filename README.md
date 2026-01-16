# Strictly Local Parameterisation of Attribution Methods  
**Decision-Boundary–Guided Baselines for Explainable AI**

This repository contains the implementation and experiments for exploring **strict locality** in attribution methods by identifying **closest decision-boundary points** and using them as **baselines** for attribution methods such as Integrated Gradients, KernelSHAP, LIME, and gradient-based saliency.

This project is part of the **MLAI Lab – Winter 2025/26**.

## Motivation

Attribution methods disagree widely due to:

- choice of baseline (e.g., zero image, noise, blurred images)  
- non-local perturbations and sampling  
- instability near sharp decision boundaries  
- the **Rashomon effect**, where multiple equally valid explanations exist  

Recent research suggests that using **decision-boundary baselines**—points lying directly on the model’s decision boundary and *closest to a sample*—leads to more **stable**, **consistent**, and **faithful** explanations.

This repository implements such baselines and evaluates their effect on attribution **locality**, **stability**, and **inter-method disagreement**.

## Core Idea

For every input sample \(x\), we compute a point \(x'\) such that:

- \(x'\) lies on the **decision boundary**,  
- the path between \(x'\) and \(x\) does **not cross other decision boundaries**,  
- the distance \(\|x - x'\|\) is minimized.  

We then compute attributions using \(x'\) as a **baseline** or **reference input**.

Algorithms implemented include:

- Informed Baseline Search (IBS) — Morasso et al., 2025  
- FGSM-based boundary search — Goodfellow et al., 2014  
- CMA-ES optimization — Nomura & Shibata, 2024  

Evaluation follows:

- Rashomon disagreement analysis — Müller et al., 2023  
- Co-12 explanation quality metrics — Nauta et al., 2023  

## Repository Structure

```
repo/
│
├── boundary_search/        # FGSM, IBS, CMA-ES-based search methods
├── attribution/            # IG, SHAP, LIME, gradients
├── evaluation/             # stability, disagreement, proximity, Co-12 metrics
├── experiments/            # scripts to run full pipelines
├── models/                 # models, training, checkpoints
├── data/                   # datasets
├── notebooks/              # exploratory notebooks and visualisations
├── results/                # saved baselines, explanations, evaluation logs
└── README.md
```

## Features

### Decision Boundary Search  
- FGSM (gradient-based)  
- IBS algorithm  
- CMA-ES for black-box search  

### Attribution Methods  
- Integrated Gradients (with custom baselines)  
- KernelSHAP (reference alternatives)  
- LIME (local perturbation baselines)  
- Saliency / Gradient × Input  

### Evaluation Suite  
- Local stability  
- Baseline sensitivity  
- Inter-method disagreement  
- Proximity metrics  
- Co-12 quantitative metrics  

## Experimental Workflow

### 1. Train a classifier  
```bash
python models/train_model.py
```

### 2. Find decision-boundary baselines  
```bash
python experiments/run_boundary_search.py
```

### 3. Compute classical & DB-based attributions  
```bash
python experiments/run_attribution.py
```

### 4. Evaluate stability, locality & disagreement  
```bash
python experiments/run_evaluation.py
```

### 5. Visualize & analyze results  
Use the notebooks in `notebooks/`.

## Installation

```bash
git clone https://github.com/UBonn-mainn/local-boundary-attribution.git
cd local-boundary-attribution

# Recommended: Conda Setup
conda create -n boundary_attribution python=3.10
conda activate boundary_attribution

# Install dependencies via Conda
conda install --file conda-requirements.txt -y
conda install pytorch torchvision -c pytorch -y

# Install remaining packages
pip install -r requirements.txt

```

## 📚 References

- Morasso et al., Informed Baseline Search, 2025  
- Goodfellow et al., FGSM, 2014  
- Nomura & Shibata, CMA-ES Python Library, 2024  
- Müller et al., Rashomon Effect in XAI, 2023  
- Nauta et al., Co-12 Quantitative Evaluation of Explainability, 2023  

## Roadmap

### Phase 1 — Boundary Search
- Implement IBS  
- Implement FGSM boundary search  
- Add CMA-ES optimizer for refinement  
- Add distance minimization objective  

### Phase 2 — Attribution
- IG with decision-boundary baseline  
- SHAP experiments with DB baselines  
- LIME experiments with DB baselines  

### Phase 3 — Evaluation
- Stability & repeatability  
- Inter-method disagreement metrics  
- Co-12 metrics  
- Final visualisation suite  
