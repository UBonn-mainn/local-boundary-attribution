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

### Our algorithm:


Evaluation follows:

- Rashomon disagreement analysis — Müller et al., 2023   

## Repository Structure

```
repo/
│
├── boundary_search/        # FGSM, Growing Sphere, BoundaryCrawler
├── attribution/            # IG, SHAP, LIME
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
- BoundaryCrawler
- Growing Sphere

### Attribution methods
- IG
- LIME
- KernelSHAP

## Experimental Workflow

### 1. Train a classifier  
```bash
python experiments/run_model.py
```

### 2. Find boundary with Growing Sphere  
```bash
python experiments/run_gs.py --root_directory <folder to find data>
```

### 3. Find decision-boundary baselines with BoundaryCrawler,   
```bash
python experiments/run.py --data_path <path-to-data>
--model_path <path-to-model>
--model_type <model-type>
--num_classes <num-classes>
--save_dir <path-to-save-results>
--vis_dir <path-to-save-visualization>
--vis_points <num-point-to-visualize>
--sphere_samples <num-sample-to-calculate-sphere-volume>
--ig_steps <IG-steps>
--topk <top-k>
--baselines <baselines-separated-by-commas>
```

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

