# MLTasks – Java Machine Learning Exercises

This repository contains a set of Java-based machine learning tasks developed for the **MPP (Machine Learning in Java)** course at Polish-Japanese Academy of Information Technology. Each task focuses on a different algorithm or concept, supported by reusable dataset-preparation and evaluation modules.

---

## 📁 Structure

Each task is located under its own package inside the `src/main/java/org/ml/` (or `com.example.ml`) directory:

- **`Main.java`** - Entry point of the program.
- **`data/`** – Utilities for parsing CSV files, splitting data into train/test sets, and encoding labels.
- **`evaluation/`** – Contains evaluation metrics (accuracy, confusion matrix, etc.).
- **`exceptions/`** - Exceptions, which may be met during the program runtime.
- **`main/`** - Main classes.
- **`models/`** - Different classifying models.
- **`plot/`** - Plotting utilities.
- **`structures/`** - Basic structures used all over the program.
- **`utils/`** – Utility needed for the program.

---

## 🔍 Topics Covered

- **Task 1: KNN (K-Nearest Neighbors)**
    - Implements a basic **KNN classifier** with a custom sorting routine and tie-breaking logic.
    - Demonstrates **multi-class classification** on the Iris dataset.
    - Explores how changing **k** affects prediction accuracy.

- **Task 2: Perceptron**
    - Implements the **Delta Rule** (gradient-based) for a single-layer perceptron.
    - Demonstrates **binary classification** on a 2D subset of the Iris dataset.
    - Shows how to plot the **decision boundary** in ASCII or export boundary points to a CSV file.

- **Task 3: Single-Layer Neural Network**
    - Implements a basic one-vs-rest neural architecture using multiple perceptrons.
    - Designed for multi-class classification of text-based language detection tasks. 
    - Each perceptron is trained to recognize one language by distinguishing it from others.
    - Supports text vectorization through normalized letter-frequency mapping (a–z).
    - Demonstrates prediction on both vectorized inputs and raw user-provided text.
  
- **Task 4: Naive Bayes Classifier**
  - Implements a categorical Naive Bayes algorithm with Laplace (add-one) smoothing.
  - Supports multi-class classification using per-feature probability caching.
  - Encodes both labels and feature values using reusable mapping structures.
  - Demonstrates classification on tabular categorical datasets (e.g., weather data).
  - Evaluates precision, recall, and F-measure for each class individually.

- **Shared Utilities**
    - **`LabelEncoder`** for converting string labels (e.g., `"setosa"`) to integers and back.
    - **`PrepareDataset`** for parsing data and performing **stratified train-test splits**.
    - **`EvaluationMetrics`** for computing accuracy and (optionally) confusion matrices.

---

## 🚀 Getting Started

1. **Clone or Download** this repository.
2. **Open in Your IDE** (e.g., IntelliJ, Eclipse) or build from the command line using Maven.
3.	Run any of the tasks:
      - Main Class: org.ml.Main (or another entry point, depending on your structure).
      - The console prompts will guide you to choose a classifier, input vectors, etc.
4.	Plotting:
      - Some tasks use ASCII plots for quick visualization.
      - Boundary points can also be exported to CSV and viewed in Excel, Python (matplotlib), or another plotting tool.

## 🛠️ Requirements
-	Java 17+ (tested with Java 17, should work with newer versions)
-	Maven (for dependency management and building)
-	No external libraries for core tasks (except optional libraries for advanced plotting)

## 🧩 How to Extend
- Add a new algorithm:
- Create a new package, e.g., org.ml.naivebayes/.
- Implement the Classifier interface or create your own.
- Use the shared dataset and evaluation modules to keep consistency.
- Add new metrics:
- Extend EvaluationMetrics with new methods (e.g., precision, recall, F1-score).
- Add advanced plotting:
- If allowed, you can integrate a library like JFreeChart or XChart for better visualizations.

## 📄 License

This project is open-source and available under the MIT License.
Feel free to modify, distribute, and use it for your own machine learning experiments.
