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
        PrepareDataset prepareDataset = new PrepareDataset();

        var dataset = prepareDataset.parseDataset("src/main/resources/csv/iris.csv", encoder, featureEncoder,false, false);
        SplitDataset splitDataset = prepareDataset.trainTestSplit(dataset, 0.86);

/*        var trainSet = prepareDataset.parseDataset("src/main/resources/csv/lang.train.csv", encoder, featureEncoder, true);
        var testSet = prepareDataset.parseDataset("src/main/resources/csv/lang.test.csv", encoder, featureEncoder, true);
        SplitDataset splitDataset = new SplitDataset(trainSet, testSet);*/

        int classesAmount = encoder.getClassesAmount();
        runKNNTests(splitDataset);
        System.out.println();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");
        System.out.println();
        runPerceptronTests(splitDataset, classesAmount);
        System.out.println();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");
        System.out.println();
        runSingleLayerNeuralNetworkTests(splitDataset, classesAmount);
        System.out.println();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");
        System.out.println();
        runNaiveBayesTests(splitDataset, classesAmount);
        System.out.println();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");
        System.out.println();

        startUserInput(splitDataset, encoder);
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

    private static void startUserInput(SplitDataset splitDataset, LabelEncoder encoder) {
        Scanner scanner = new Scanner(System.in);
        boolean exit = true;
        while (exit) {
            System.out.println("Possible actions:");
            System.out.println("1. Evaluate new vector");
            System.out.println("2. Exit");
            System.out.println("Enter your choice: ");

            switch (scanner.nextLine()) {
                case "1" -> predictFromUserInput(splitDataset, scanner, encoder);
                case "2" -> exit = false;
                default -> System.out.println("Unknown action. Try again.");
            }
        }

        scanner.close();
    }

    private static void predictFromUserInput(SplitDataset splitDataset,
                                             Scanner scanner,
                                             LabelEncoder encoder) {
        System.out.println("Choose a classifier:");
        System.out.println("  1) KNN");
        System.out.println("  2) Perceptron");
        System.out.println("  3) Single Layer Neural Network");
        System.out.println("  4) Naive Bayes");
        String choice = scanner.nextLine().trim();

        Classifier chosen;
        int classesAmount = encoder.getClassesAmount();

        switch (choice) {
            case "1": {
                // --- KNN parameters ---
                System.out.print("Enter k (number of neighbors): ");
                int k = Integer.parseInt(scanner.nextLine().trim());
                KNearestNeighbours knn = new KNearestNeighbours();
                knn.setK(k);
                chosen = knn;
                break;
            }
            case "2": {
                // --- Perceptron parameters ---
                System.out.print("Enter learning rate α (e.g. 0.1): ");
                double alpha = Double.parseDouble(scanner.nextLine().trim());
                chosen = new Perceptron(alpha);
                break;
            }
            case "3": {
                // --- Single Layer NN parameters ---
                System.out.print("Enter learning rate α (e.g. 0.01): ");
                double alpha3 = Double.parseDouble(scanner.nextLine().trim());
                System.out.print("Enter momentum β (e.g. 0.0): ");
                double beta = Double.parseDouble(scanner.nextLine().trim());
                chosen = new SingleLayerNeuralNetwork(alpha3, beta, classesAmount);
                break;
            }
            case "4": {
                // --- Naive Bayes parameters ---
                System.out.print("Apply Laplace smoothing to all counts? (y/n): ");
                String yn = scanner.nextLine().trim().toLowerCase();
                boolean smoothAll = yn.startsWith("y");
                chosen = new NaiveBayes(classesAmount, smoothAll);
                break;
            }
            default: {
                System.err.println("Invalid selection, defaulting to KNN with k=3.");
                KNearestNeighbours knn = new KNearestNeighbours();
                knn.setK(3);
                chosen = knn;
            }
        }

        // Train it once on your split
        chosen.train(splitDataset.getTrainSet());

        // Now read the test vector
        System.out.println("Enter '1' to type a numeric vector or '2' for raw text:");
        String mode = scanner.nextLine().trim();

        double[] features;
        if (mode.equals("2")) {
            System.out.println("Enter your text:");
            String text = scanner.nextLine();
            features = PrepareDataset.textToVector(text);
        } else {
            System.out.println("Enter comma‑separated numbers:");
            String[] tokens = scanner.nextLine().split(",");
            features = new double[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                features[i] = Double.parseDouble(tokens[i].trim());
            }
        }

        // Predict and decode
        int raw = chosen.predict(features);
        String decoded = encoder.decode(raw);
        System.out.println("→ Predicted class index: " + raw);
        System.out.println("→ Predicted label:       " + decoded);

        // If it's the Perceptron, show boundary
        if (chosen instanceof Perceptron p) {
            DecisionBoundaryPlotter plotter =
                    new DecisionBoundaryPlotter(p, 0.0, 10.0, 50);
            System.out.println("ASCII decision boundary:");
            plotter.asciiPlot(splitDataset.getTestSet());
            plotter.exportBoundaryToCsv("src/main/resources/boundary.csv");
        }
    }
}