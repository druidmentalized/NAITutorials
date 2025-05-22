# MLTasks – Java Machine Learning Exercises

This repository contains a set of Java-based machine learning tasks developed for the **MPP (Machine Learning in Java)** course at the Polish-Japanese Academy of Information Technology. Each task focuses on a different algorithm or concept, supported by reusable dataset-preparation, evaluation, and clustering modules.

---

## 📁 Structure

Each component is organized under its own package in `src/main/java/org/nai/`:
- **`Main.java`** – Entry point of the program.
- **`algorithms/`** – Standalone algorithms such as the **Knapsack Problem**.
- **`data/`** – Utilities for parsing CSVs, splitting into train/test sets, and encoding labels.
- **`evaluation/`** – Evaluation metrics (accuracy, precision, recall, F1, WCSS/RSS).
- **`exceptions/`** – Custom runtime exceptions.
- **`models/`** – Classification models (KNN, Perceptron, Neural Network, Naive Bayes) and the **KMeansClusterer**.
- **`plot/`** – Visualization tools (decision boundaries, cluster scatter plots, WCSS plots).
- **`structures/`** – Fundamental structures (Pair, Triple, Vector, Cluster, Centroid).
- **`utils/`** – Miscellaneous helpers.

---

## 🔍 Topics Covered

### Task 1: KNN (K-Nearest Neighbors)
- Implements a basic **KNN classifier** with sorting and tie-breaking.
- Supports **multi-class classification** (e.g., Iris).
- Parameter sweep over **k** values.

### Task 2: Perceptron
- Implements **Delta Rule** for a single-layer binary classifier.
- ASCII/CSV-based decision boundary visualization.

### Task 3: Single-Layer Neural Network
- One-vs-rest perceptrons for **text language classification**.
- Uses **letter-frequency vectorization** of raw text.

### Task 4: Naive Bayes Classifier
- **Categorical Naive Bayes** with optional Laplace smoothing.
- Per-class evaluation (accuracy, precision, recall, F1).

### Task 5: K-Means Clustering
- Classic **K-Means** unsupervised clustering for numeric features.
- Random initialization, iterative reassignment, and centroid recomputation.
- Measures **WCSS** for elbow method.
- Produces multi-dimensional cluster plots and WCSS–k curve.

### Task 6: Knapsack Problem (Brute Force & Greedy)
- Solves **0/1 Knapsack** via:
  - **Brute-force**: evaluates all subsets for max value under capacity.
  - **Greedy (density-based)**: picks items by decreasing value-to-weight ratio.
- Outputs the best subset of item weights and value sum.
- Measures and compares runtimes between methods.

### Task 7: Knapsack Problem (Hill Climbing)
- Implements **Hill Climbing with random restarts** for the 0/1 Knapsack problem.
- Each run starts from a randomly selected item and greedily adds available items to improve the solution.
- Tracks the best solution across multiple restarts.
- Outputs the final best subset of weights and its total value.
- Compares runtime and solution quality against Brute-force and Greedy approaches.

---

## Shared Utilities

- **`LabelEncoder`**, **`FeatureEncoder`**: Encode string labels or categorical attributes as integers.
- **`PrepareDataset`**: CSV/Folder parsing + stratified **train-test split**.
- **`EvaluationMetrics`**: Accuracy, precision, recall, F1 for classifiers + WCSS for clustering.

---

## 🚀 Getting Started

1. Clone this repository.
2. Open in IntelliJ or Eclipse, or build via Maven:
   ```bash
   mvn clean install
3. Run:
  - **Main class**: org.nai.main

## 🛠️ Requirements

- Java 17+ (tested on Java 17, compatible with newer).
- Maven (for build and dependencies).
- Core algorithms use **only base Java** libraries.
- **XChart** or **JFreeChart** may be used for plotting (optional).

---

## 🧩 How to Extend

- **Add a new classifier or clusterer**:

  1. Create a new Model (implements Classifier) or Clusterer interface.
  2. Plug into Main via chooseClassifier or runClusterersTests.
  3. Use PrepareDataset and EvaluationMetrics to avoid boilerplate.

- **Add advanced plots**:

  * Integrate XChart/JFreeChart in plot/.

---

## 📄 License

This project is MIT-licensed. Feel free to modify and distribute for your machine learning experiments.
