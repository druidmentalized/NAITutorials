# MLTasks – Java Machine Learning Exercises

This repository contains a set of Java-based machine learning tasks developed for the **MPP (Machine Learning in Java)** course at the Polish-Japanese Academy of Information Technology. Each task focuses on a different algorithm or concept, supported by reusable dataset-preparation, evaluation, and **clustering** modules.

---

## 📁 Structure

Each component is organized under its own package in `src/main/java/org/ml/`:
- **`Main.java`** – Entry point of the program.
- **`data/`** – Utilities for parsing CSVs, splitting into train/test sets, and encoding labels.
- **`evaluation/`** – Contains evaluation metrics (accuracy, precision, recall, F-measure, and clustering metrics such as WCSS/RSS).
- **`exceptions/`** – Custom runtime exceptions.
- **`models/`** – Classification models (KNN, Perceptron, Neural Network, Naive Bayes) and the **KMeansClusterer** clustering model.
- **`plot/`** – Plotting utilities for decision boundaries, WCSS charts, and cluster visualizations.
- **`structures/`** – Fundamental data structures (Pair, Triple, Vector, Cluster, Centroid).
- **`utils/`** – Miscellaneous helpers (distance calculations, file readers, etc.).

---

## 🔍 Topics Covered

- **Task 1: KNN (K-Nearest Neighbors)**

  * Implements a basic **KNN classifier** with custom sorting and tie-breaking.
  * Demonstrates **multi-class** classification on the Iris dataset.
  * Explores how changing **k** affects accuracy.

- **Task 2: Perceptron**

  - Implements the **Delta Rule** (gradient ascent) for a single-layer perceptron.
  - Demonstrates **binary classification** in 2D and ASCII/CSV decision boundary plotting.

- **Task 3: Single-Layer Neural Network**

  - One-vs-rest architecture using multiple Perceptrons for **text language detection**.
  - Vectorizes text via normalized letter-frequency (a–z) mappings.

- **Task 4: Naive Bayes Classifier**

  - Categorical Naive Bayes with optional **Laplace smoothing**.
  - Multi-class classification with per-feature probability caching.
  - Evaluates **accuracy**, **precision**, **recall**, and **F-measure** per class.

- **Task 5: K-Means Clustering**

  - Implements an **unsupervised** K-Means algorithm (`KMeansClusterer`) for numeric data.
  - **Parameters**: `k` (number of clusters).
  - **Initialization**: random assignment of points to clusters, ensuring no empty cluster.
  - **Iteration**: re-assign points to nearest centroid, recompute centroids, repeat until convergence.
  - **Evaluation**: computes **WCSS/RSS** to measure cluster compactness.
  - **Visualization**: scatter-matrix of all attribute pairs colored by cluster, centroids highlighted; WCSS vs. `k` elbow chart.

- **Shared Utilities**

  - **`LabelEncoder`** – Map string labels ↔ integers.
  - **`FeatureEncoder`** – Convert categorical feature values ↔ integers.
  - **`PrepareDataset`** – Stratified **train/test split** and data parsing.
  - **`EvaluationMetrics`** – Classification metrics and **WCSS** for clustering.

---

## 🚀 Getting Started

1. **Clone** this repository.
2. **Open** in your IDE (IntelliJ/Eclipse) or build via Maven:

   ```bash
   mvn clean install
   ```
3. **Run** the main class (`org.ml.Main`):

  * Choose a dataset, a classifier or clusterer, and view results.
  * Follow console prompts for parameters (k, α, β, smoothing).

---

## 🛠️ Requirements

- Java 17+ (tested on Java 17, compatible with newer).
- Maven (for build and dependencies).
- Core algorithms use **only base Java** libraries.
- **XChart** or **JFreeChart** may be used for plotting (optional).

---

## 🧩 How to Extend

- **Add a new classifier or clusterer**:

  1. Create a new `Model` (implements `Classifier`) or `Clusterer` interface.
  2. Plug into `Main` via `chooseClassifier` or `runClusterersTests`.
  3. Use `PrepareDataset` and `EvaluationMetrics` to avoid boilerplate.

- **Add advanced plots**:

  * Integrate XChart/JFreeChart in `plot/`.

---

## 📄 License

This project is MIT-licensed. Feel free to modify and distribute for your machine learning experiments.
