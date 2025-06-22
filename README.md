# MNIST Digit Classification with k-Nearest Neighbors (k-NN)

Analyze and classify handwritten digits from the MNIST dataset using a vanilla **k-NN** classifier, then explore how training-set size, data-splitting strategy, and distance metric affect accuracy.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Setup & Installation](#setup--installation)  
4. [Project Structure](#project-structure)  
5. [Running the Experiments](#running-the-experiments)  
6. [Key Results](#key-results)  
7. [Discussion](#discussion)  
8. [Contributing](#contributing)  
9. [License](#license)  
10. [References](#references)

---

## Project Overview
This repository contains code and documentation for evaluating **k-Nearest Neighbors** on six MNIST digits (0, 1, 2, 3, 6, 7).  
The goals are to:

- Estimate computational cost for the full dataset  
- Identify the smallest training size that stabilizes accuracy  
- Compare three data-splitting methods  
- Benchmark Euclidean vs. Manhattan distance  
- Produce per-class precision, recall, and F1 to spot hard-to-classify digits

The best configuration reaches **95.64%** test accuracy with **6,000** training samples, **cluster-based sampling**, and **Euclidean distance**.

---

## Dataset
| Split | Samples | Description |
|-------|---------|-------------|
| Train | 36,937  | Filtered MNIST digits 0, 1, 2, 3, 6, 7 (28×28 px, flattened → 784 features) |
| Test  | 6,143   | Same digits, never seen during training |

Pre-processing steps:
- IDX → NumPy conversion  
- Flatten images to one-dimensional vectors  
- Min-max normalization to [0, 1]

---

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/mnist-knn-analysis.git
cd mnist-knn-analysis

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt
```

Python ≥ 3.9 and `scikit-learn`, `numpy`, `pandas`, `matplotlib` are required.

---

## Project Structure

```
mnist-knn-analysis/
│
├─ data/                 # Downloaded IDX files or .npz cache
├─ notebooks/
│   └─ mnist_knn.ipynb    # Jupyter notebook with all experiments
├─ src/
│   ├─ dataset.py         # Loading & preprocessing helpers
│   ├─ sampling.py        # Random, stratified & cluster-based splits
│   ├─ knn_evaluate.py    # Training, timing & metric utilities
│   └─ plots.py           # Optional visualization helpers
├─ results/
│   ├─ timings.txt
│   ├─ accuracy_vs_train_size.png
│   └─ classification_report.csv
├─ requirements.txt
└─ README.md              # You are here
```

---

## Running the Experiments

All steps may be executed inside the notebook or as a script:

```bash
python src/knn_evaluate.py \
  --digits 0 1 2 3 6 7 \
  --train-size 6000 \
  --split-method cluster \
  --distance euclidean \
  --k 20
```

Flags:

- `--train-size {N}`: subsample N training points  
- `--split-method {random|stratified|cluster}`  
- `--distance {euclidean|manhattan}`  
- `--k {int}`: number of neighbors

---

## Key Results

| Experiment                                 | Accuracy |
|--------------------------------------------|----------|
| **Best model** (6,000 train, cluster, L2)  | **95.64%** |
| Random split (6,000 train, L2)             | 95.56% |
| Stratified split (6,000 train, L2)         | 95.56% |
| Manhattan distance (6,000 train, cluster)  | 94.81% |

### Per-class metrics (best model):

| Digit | Precision | Recall | F1 |
|-------|-----------|--------|----|
| 0     | 96.90%    | 98.78% | 97.83% |
| 1     | 89.18%    | 99.47% | 94.04% |
| **2** | 98.80%    | **87.69%** | 92.92% |
| 3     | 98.88%    | 95.94% | 97.39% |
| 6     | 97.60%    | 97.70% | 97.65% |
| 7     | 94.62%    | 94.16% | 94.39% |

Digit **2** is the hardest to classify, likely due to shape overlap with other curved digits.

---

## Discussion

- **Training-set size**: performance flattens beyond **6,000** samples, offering a good speed/accuracy trade-off.  
- **Splitting strategy**: clustering ensures diverse prototypes, giving a slight edge.  
- **Distance metric**: Euclidean (L2) consistently outperforms Manhattan (L1) on pixel data.  
- **Compute**: end-to-end runtime ≈ 23s on a modern CPU; fits comfortably in <1 GB RAM.

Feel free to open issues or pull requests with improvements—e.g., dimensionality-reduction, distance-weighted voting, or GPU-accelerated neighbors.

---

## Contributing

1. Fork the project  
2. Create a feature branch: `git checkout -b feat/awesome-idea`  
3. Commit your changes: `git commit -m 'Add awesome idea'`  
4. Push to the branch: `git push origin feat/awesome-idea`  
5. Open a Pull Request

---

## License

MIT License – see [`LICENSE`](LICENSE) for details.

---

## References

- Yann LeCun et al., *“The MNIST Database of Handwritten Digits”*  
- Scikit-Learn documentation – `sklearn.neighbors.KNeighborsClassifier`

> *“Simplicity is the ultimate sophistication.”* – Leonardo da Vinci
