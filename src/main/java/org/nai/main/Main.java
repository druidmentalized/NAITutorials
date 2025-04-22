package org.nai.main;

import org.nai.data.PrepareDataset;
import org.nai.data.SplitDataset;
import org.nai.evaluation.EvaluationMetrics;
import org.nai.models.*;
import org.nai.plot.DecisionBoundaryPlotter;
import org.nai.utils.FeatureEncoder;
import org.nai.utils.LabelEncoder;

import java.util.*;

public class Main {
    public static void main(String[] args) {
        LabelEncoder encoder = new LabelEncoder();
        FeatureEncoder featureEncoder = new FeatureEncoder();
        PrepareDataset prepare = new PrepareDataset();

        // Preload all datasets
        Map<String, SplitDataset> datasets = new LinkedHashMap<>();
        datasets.put("1) Iris (numeric)",
                prepare.trainTestSplit(
                        prepare.parseDataset("src/main/resources/csv/iris.csv", encoder, featureEncoder, false, false),
                        0.66));
        datasets.put("2) outGame (categorical)",
                prepare.trainTestSplit(
                        prepare.parseDataset("src/main/resources/csv/outGame.csv", encoder, featureEncoder, false, true),
                        0.86));
        datasets.put("3) Language CSVs (text)",
                new SplitDataset(
                        prepare.parseDataset("src/main/resources/csv/lang.train.csv", encoder, featureEncoder, true, false),
                        prepare.parseDataset("src/main/resources/csv/lang.test.csv", encoder, featureEncoder, true, false)
                ));
        datasets.put("4) Languages folder (text)",
                prepare.trainTestSplit(
                        prepare.parseDataset("src/main/resources/languagesdataset", encoder, featureEncoder, false, false),
                        0.66));

        // Choose dataset
        SplitDataset current = chooseDataset(datasets);
        int classesAmount = encoder.getClassesAmount();

        // Run all tests
        //runKNNTests(current);
        divider();
        runPerceptronTests(current, classesAmount);
        divider();
        runSingleLayerNeuralNetworkTests(current, classesAmount);
        divider();
        runNaiveBayesTests(current, classesAmount);
        divider();

        // Interactive prediction loop
        startUserInput(current, encoder);
    }

    private static SplitDataset chooseDataset(Map<String, SplitDataset> datasets) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Select dataset:");
        for (String name : datasets.keySet()) {
            System.out.println(name);
        }
        System.out.print("Enter choice number: ");
        String choice = sc.nextLine().trim();
        int idx = Math.max(1, Math.min(datasets.size(), Integer.parseInt(choice)));
        return new ArrayList<>(datasets.values()).get(idx - 1);
    }

    private static void divider() {
        System.out.println("\n────────────────────────────────────────────────────────────────────────────────────\n");
    }

    private static void runKNNTests(SplitDataset splitDataset) {
        KNearestNeighbours knn = new KNearestNeighbours();
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics(knn, splitDataset);

        System.out.println("Testing of the KNN algorithm\n");
        System.out.println("Testcase 1, k=3:");
        knn.setK(3);
        evaluationMetrics.measureAccuracy();
        System.out.println();

        System.out.println("Testcase 2, k=7:");
        knn.setK(7);
        evaluationMetrics.measureAccuracy();
        System.out.println();

        System.out.println("Testcase 3, k=11:");
        knn.setK(11);
        evaluationMetrics.measureAccuracy();
        System.out.println();

        System.out.println("Testcase 4, k=15:");
        knn.setK(15);
        evaluationMetrics.measureAccuracy();
        System.out.println();

        System.out.println("Testcase 5, k=20:");
        knn.setK(20);
        evaluationMetrics.measureAccuracy();
    }

    private static void runPerceptronTests(SplitDataset splitDataset, int classesAmount) {
        System.out.println("Testing of the Perceptron algorithm\n");
        Perceptron perceptron = new Perceptron(0.2);
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics(perceptron, splitDataset);
        outputEvaluations(evaluationMetrics, classesAmount);
    }

    private static void runSingleLayerNeuralNetworkTests(SplitDataset splitDataset, int classesAmount) {
        System.out.println("Testing of the Single Layer Neural Network algorithm\n");
        SingleLayerNeuralNetwork singleLayerNeuralNetwork = new SingleLayerNeuralNetwork(0.01, 0.0, classesAmount);
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics(singleLayerNeuralNetwork, splitDataset);
        outputEvaluations(evaluationMetrics, classesAmount);
    }

    private static void runNaiveBayesTests(SplitDataset splitDataset, int classesAmount) {
        System.out.println("Testing of the Naive Bayes Network algorithm\n");
        NaiveBayes naiveBayes = new NaiveBayes(classesAmount, true);
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics(naiveBayes, splitDataset);
        outputEvaluations(evaluationMetrics, classesAmount);
    }

    private static void outputEvaluations(EvaluationMetrics evaluationMetrics, int classesAmount) {
        evaluationMetrics.measureAccuracy();
        for (int i = 0; i < classesAmount; i++) {
            System.out.println("\nClass " + i + " measuring:");
            double precision = evaluationMetrics.evaluatePrecision(i);
            double recall = evaluationMetrics.evaluateRecall(i);
            evaluationMetrics.evaluateF1Measure(precision, recall);
        }
    }

    private static void startUserInput(SplitDataset split, LabelEncoder encoder) {
        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.println("\nActions: [1] Predict new sample   [2] Exit");
            System.out.print("Choice: ");
            String cmd = sc.nextLine().trim();
            if (cmd.equals("2")) break;
            if (cmd.equals("1")) predictFromUserInput(split, sc, encoder);
        }
    }

    private static void predictFromUserInput(SplitDataset split, Scanner sc, LabelEncoder encoder) {
        Classifier clf = chooseClassifier(sc, encoder.getClassesAmount());
        clf.train(split.getTrainSet());

        double[] features = readFeatures(sc);
        int label = clf.predict(features);
        System.out.println("Predicted index: " + label + ", label: " + encoder.decode(label));

        if (clf instanceof Perceptron p) {
            DecisionBoundaryPlotter plot = new DecisionBoundaryPlotter(p, 0, 10, 50);
            plot.asciiPlot(split.getTestSet());
            plot.exportBoundaryToCsv("src/main/resources/boundary.csv");
        }
    }

    private static Classifier chooseClassifier(Scanner sc, int classesAmount) {
        System.out.println("Select classifier: 1)KNN 2)Percep 3)SLNN 4)Bayes");
        String c = sc.nextLine().trim();
        return switch (c) {
            case "1" -> {
                System.out.print("k? ");
                int k = Integer.parseInt(sc.nextLine().trim());
                KNearestNeighbours kNN = new KNearestNeighbours(); kNN.setK(k);
                yield kNN;
            }
            case "2" -> {
                System.out.print("alpha? ");
                double a = Double.parseDouble(sc.nextLine().trim());
                yield new Perceptron(a);
            }
            case "3" -> {
                System.out.print("alpha? "); double a = Double.parseDouble(sc.nextLine().trim());
                System.out.print("beta?  "); double b = Double.parseDouble(sc.nextLine().trim());
                yield new SingleLayerNeuralNetwork(a, b, classesAmount);
            }
            case "4" -> {
                System.out.print("smooth all? (y/n): ");
                boolean s = sc.nextLine().toLowerCase().startsWith("y");
                yield new NaiveBayes(classesAmount, s);
            }
            default -> {
                System.err.println("Invalid, using KNN(k=3)");
                KNearestNeighbours def = new KNearestNeighbours(); def.setK(3);
                yield def;
            }
        };
    }

    private static double[] readFeatures(Scanner sc) {
        System.out.print("Input type: 1)numeric 2)text: ");
        if (sc.nextLine().trim().equals("2")) {
            System.out.print("Enter text: ");
            return PrepareDataset.textToVector(sc.nextLine());
        }
        System.out.print("Enter comma vals: ");
        String[] toks = sc.nextLine().split(",");
        double[] f = new double[toks.length];
        for (int i = 0; i < toks.length; i++) f[i] = Double.parseDouble(toks[i].trim());
        return f;
    }
}
