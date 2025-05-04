package org.nai.main;

import org.nai.data.PrepareDataset;
import org.nai.data.SplitDataset;
import org.nai.data.Dataset;
import org.nai.evaluation.EvaluationMetrics;
import org.nai.models.*;
import org.nai.models.KMeansClusterer;
import org.nai.plot.KMeansClustersPlotter;
import org.nai.structures.Cluster;
import org.nai.structures.Pair;
import org.nai.structures.Vector;
import org.nai.utils.FeatureEncoder;
import org.nai.utils.LabelEncoder;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        runTests();
    }

    private static void runTests() {
        PrepareDataset prepare = new PrepareDataset();

        LabelEncoder irisEncoder = new LabelEncoder();
        FeatureEncoder irisFe = new FeatureEncoder();
        Dataset irisDataset = prepare.parseDataset(
            "src/main/resources/csv/iris.csv",
            irisEncoder, irisFe,
            false, false
        );
        SplitDataset irisSplit = prepare.trainTestSplit(irisDataset, 0.66);

        // Run classifiers on the split data
        runClassifiersTests(irisSplit, irisEncoder, irisFe);
        divider();
        // Run clustering on full data
        runClusterersTests(irisDataset);
        divider();

        // Start interactive input on the split data
        startUserInput(irisSplit, irisEncoder);
    }

    private static void runClassifiersTests(SplitDataset splitDataset,
                                            LabelEncoder encoder,
                                            FeatureEncoder fe) {
        SplitDataset current = splitDataset;
        int classesAmount = encoder.getClassesAmount();

        // Run all tests
        runKNNTests(current);
        divider();
        runPerceptronTests(current, classesAmount);
        divider();
        runSingleLayerNeuralNetworkTests(current, classesAmount);
        divider();
        runNaiveBayesTests(current, classesAmount);
        divider();
    }

    private static void runClusterersTests(Dataset dataset) {
        List<Vector> vectors = dataset.getVectors();

        runKMeansClustererTests(vectors);
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
        outputClassifierEvaluations(evaluationMetrics, classesAmount);
    }

    private static void runSingleLayerNeuralNetworkTests(SplitDataset splitDataset, int classesAmount) {
        System.out.println("Testing of the Single Layer Neural Network algorithm\n");
        SingleLayerNeuralNetwork singleLayerNeuralNetwork = new SingleLayerNeuralNetwork(0.01, classesAmount);
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics(singleLayerNeuralNetwork, splitDataset);
        outputClassifierEvaluations(evaluationMetrics, classesAmount);
    }

    private static void runNaiveBayesTests(SplitDataset splitDataset, int classesAmount) {
        System.out.println("Testing of the Naive Bayes Network algorithm\n");
        NaiveBayes naiveBayes = new NaiveBayes(classesAmount, true);
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics(naiveBayes, splitDataset);
        outputClassifierEvaluations(evaluationMetrics, classesAmount);
    }

    private static void runKMeansClustererTests(List<Vector> vectors) {
        System.out.println("Testing of the K-Means Clustering algorithm\n");

        KMeansClusterer kMeansClusterer = new KMeansClusterer();

        double bestWcss = 0;
        double bestWcssDiff = Double.MIN_VALUE;
        double prevWcss = Double.MIN_VALUE;
        List<Cluster> bestClusters = null;
        int bestK = 2;

        List<Pair<Integer, Double>> kToWcss = new ArrayList<>();

        for (int k = 2; k <= 15; k++) {
            List<Cluster> clusters = kMeansClusterer.groupClusters(k, vectors);
            double wcss = EvaluationMetrics.computeWCSS(clusters);

            System.out.printf("k = %d → WCSS = %.4f\n", k, wcss);

            kToWcss.add(new Pair<>(k, wcss));

            double wcssDiff = prevWcss - wcss;
            if (wcssDiff > bestWcssDiff) {
                bestWcssDiff = wcssDiff;
                bestClusters = clusters;
                bestWcss = wcss;
                bestK = k;
            }

            prevWcss = wcss;
        }

        System.out.printf("\nBest clustering found at k = %d with WCSS = %.4f\n", bestK, bestWcss);

        KMeansClustersPlotter.plotWCSS(kToWcss);

        if (bestClusters != null) {
            KMeansClustersPlotter.plotClusters(bestClusters);
        }
    }

    private static void outputClassifierEvaluations(EvaluationMetrics evaluationMetrics, int classesAmount) {
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
            if (cmd.equals("2")) System.exit(0);
            if (cmd.equals("1")) predictFromUserInput(split, sc, encoder);
        }
    }

    private static void predictFromUserInput(SplitDataset split, Scanner sc, LabelEncoder encoder) {
        Classifier classifier = chooseClassifier(sc, encoder.getClassesAmount());
        classifier.train(split.getTrainSet());

        Vector features = readFeatures(sc);
        int label = classifier.predict(features);
        System.out.println("Predicted index: " + label + ", label: " + encoder.decode(label));
    }

    private static Classifier chooseClassifier(Scanner sc, int classesAmount) {
        System.out.println("Select classifier: 1)KNN 2)Perceptron 3)Single-Layer Neuron Network 4)Bayes");
        String c = sc.nextLine().trim();
        return switch (c) {
            case "1" -> {
                System.out.print("k? ");
                int k = Integer.parseInt(sc.nextLine().trim());
                KNearestNeighbours kNN = new KNearestNeighbours();
                kNN.setK(k);
                yield kNN;
            }
            case "2" -> {
                System.out.print("alpha? ");
                double a = Double.parseDouble(sc.nextLine().trim());
                yield new Perceptron(a);
            }
            case "3" -> {
                System.out.print("alpha? ");
                double a = Double.parseDouble(sc.nextLine().trim());
                yield new SingleLayerNeuralNetwork(a, classesAmount);
            }
            case "4" -> {
                System.out.print("smooth all? (y/n): ");
                boolean s = sc.nextLine().toLowerCase().startsWith("y");
                yield new NaiveBayes(classesAmount, s);
            }
            default -> {
                System.err.println("Invalid, using KNN(k=3)");
                KNearestNeighbours def = new KNearestNeighbours();
                def.setK(3);
                yield def;
            }
        };
    }

    private static Vector readFeatures(Scanner sc) {
        System.out.print("Input type: 1)numeric 2)text: ");
        if (sc.nextLine().trim().equals("2")) {
            System.out.print("Enter text: ");
            return PrepareDataset.textToVector(sc.nextLine());
        }
        System.out.print("Enter comma vals: ");
        String[] tokens = sc.nextLine().split(",");
        double[] f = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) f[i] = Double.parseDouble(tokens[i].trim());
        return new Vector(f);
    }
}
