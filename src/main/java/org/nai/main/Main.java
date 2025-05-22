package org.nai.main;

import org.nai.algorithms.Knapsack;
import org.nai.data.Dataset;
import org.nai.data.PrepareDataset;
import org.nai.data.SplitDataset;
import org.nai.evaluation.EvaluationMetrics;
import org.nai.models.*;
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
// *       runClassifiersTests(irisSplit, irisEncoder);
        divider();
        // Run clustering on full data
// *      runClusterersTests(irisDataset);
        divider();
        // Run Algorithms on Vectors
        runAlgorithmsTests(irisDataset.getVectors());
        divider();

        // Start interactive input on the split data
        startUserInput(irisSplit, irisEncoder);
    }

    private static void runClassifiersTests(SplitDataset splitDataset,
                                            LabelEncoder encoder) {
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

    private static void runAlgorithmsTests(List<Vector> inputVectors) {
        // * Basic values test
        Vector itemWeights = new Vector(new double[]{8, 10, 3, 5, 2});
        Vector itemValues = new Vector(new double[]{10, 12, 5, 6, 2});
        double capacity = 10;
        runKnapsackTests(itemWeights, itemValues, capacity);

        System.out.println("\n Bigger values test:");

        itemWeights = new Vector(new double[]{5, 12, 7, 18, 3, 14, 20, 1, 9, 16, 11, 8});
        itemValues = new Vector(new double[]{10, 24, 15, 40, 7, 28, 50, 5, 18, 36, 30, 20});
        capacity = 50;

        runKnapsackTests(itemWeights, itemValues, capacity);

        System.out.println("\nTeacher's values test:");
        itemWeights = new Vector(new double[]{5, 3, 9, 2, 11, 4, 4, 10, 2, 6, 6, 2, 2, 1, 1, 3, 11, 8, 8, 7, 8, 8, 8, 7, 1, 2, 10, 2});
        itemValues = new Vector(new double[]{4, 8, 43, 60, 63, 29, 44, 27, 95, 51, 22, 78, 85, 97, 64, 93, 31, 28, 22, 63, 89, 5, 98, 27, 63, 49, 31, 93});
        capacity = 12;

        runKnapsackTests(itemWeights, itemValues, capacity);
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

    private static void runKnapsackTests(Vector itemWeights, Vector itemValues, double capacity) {
        System.out.println("Running Knapsack test with:");
        System.out.println("Item Weights: " + itemWeights);
        System.out.println("Item Values: " + itemValues);
        System.out.println("Capacity = " + capacity + "\n");

        Knapsack knap = new Knapsack(itemWeights, itemValues, capacity);

        //Hill climbing timing
        long startHill = System.nanoTime();
        Pair<Vector, Double> hillRes = knap.hillClimbing(7);
        long timeHill = System.nanoTime() - startHill;

        // Print
        System.out.printf("Hill climbing: value=%.0f, weights=%s, time=%,dms%n",
                hillRes.second(),
                hillRes.first(),
                timeHill / 1_000);

        // Greedy timing
        long startGreedy = System.nanoTime();
        Pair<Vector, Double> greedyRes = knap.greedyDensityApproach();
        long timeGreedy = System.nanoTime() - startGreedy;

        System.out.printf("Greedy-density: value=%.0f, weights=%s, time=%,dms%n",
                greedyRes.second(),
                greedyRes.first(),
                timeGreedy / 1_000);


        // Brute-force timing
        long startBrute = System.nanoTime();
        Pair<List<Vector>, Double> bruteRes = knap.bruteForce();
        long timeBrute = System.nanoTime() - startBrute;

        System.out.printf("Brute-force: value=%.0f, weights=%s, time=%,dms%n",
                bruteRes.second(),
                bruteRes.first(),
                timeBrute / 1_000);
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

            System.out.printf("k = %d → WCSS = %.4f%n", k, wcss);

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

        System.out.printf("%nBest clustering found at k = %d with WCSS = %.4f%n", bestK, bestWcss);

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
        int inputs = 0;
        while (inputs < 1000) {
            System.out.println("\nActions: [1] Predict new sample   [2] Exit");
            System.out.print("Choice: ");
            String cmd = sc.nextLine().trim();
            if (cmd.equals("2")) System.exit(0);
            if (cmd.equals("1")) predictFromUserInput(split, sc, encoder);

            inputs++;
        }
    }

    private static void predictFromUserInput(SplitDataset split, Scanner sc, LabelEncoder encoder) {
        Classifier classifier = chooseClassifier(sc, encoder.getClassesAmount());
        classifier.train(split.trainSet());

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
