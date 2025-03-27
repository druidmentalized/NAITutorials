package org.knn.main;

import org.knn.data.PrepareDataset;
import org.knn.data.SplitDataset;
import org.knn.evaluation.EvaluationMetrics;
import org.knn.models.Classifier;
import org.knn.models.KNearestNeighbours;
import org.knn.models.Perceptron;
import org.knn.models.SingleLayerNeuralNetwork;
import org.knn.plot.DecisionBoundaryPlotter;
import org.knn.structures.Pair;
import org.knn.utils.LabelEncoder;

import java.util.*;

public class Main {
    public static void main(String[] args) {
        LabelEncoder encoder = new LabelEncoder();
        PrepareDataset prepareDataset = new PrepareDataset();
        var dataset = prepareDataset.parseDataset("src/main/resources/languagesdataset", encoder);
        SplitDataset splitDataset = prepareDataset.trainTestSplit(dataset, 0.66);

        //runKNNTests(splitDataset);
        //System.out.println();
        //runPerceptronTests(splitDataset);
        System.out.println();
        int classesAmount = encoder.getClassesAmount();
        runSingleLayerNeuralNetworkTests(splitDataset, classesAmount);


        //startUserInput(splitDataset, encoder);
    }

    private static void runKNNTests(SplitDataset splitDataset) {
        KNearestNeighbours knn = new KNearestNeighbours();
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics(knn, splitDataset);

        System.out.println("Testing of the KNN algorithm");
        System.out.println("Testcase 1, k=3:");
        knn.setK(3);
        evaluationMetrics.measureAccuracy();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");

        System.out.println("Testcase 2, k=7:");
        knn.setK(7);
        evaluationMetrics.measureAccuracy();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");

        System.out.println("Testcase 3, k=11:");
        knn.setK(11);
        evaluationMetrics.measureAccuracy();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");

        System.out.println("Testcase 4, k=15:");
        knn.setK(15);
        evaluationMetrics.measureAccuracy();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");

        System.out.println("Testcase 5, k=20:");
        knn.setK(20);
        evaluationMetrics.measureAccuracy();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");
    }

    private static void runPerceptronTests(SplitDataset splitDataset) {
        Perceptron perceptron = new Perceptron(0.5);
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics(perceptron, splitDataset);
        System.out.println("Testing of the Perceptron algorithm");
        evaluationMetrics.measureAccuracy();
        double precision = evaluationMetrics.evaluatePrecision(splitDataset.getTestSetLabels().getFirst());
        double recall = evaluationMetrics.evaluateRecall(splitDataset.getTestSetLabels().getFirst());
        evaluationMetrics.evaluateF1Measure(precision, recall);
    }

    private static void runSingleLayerNeuralNetworkTests(SplitDataset splitDataset, int classesAmount) {
        SingleLayerNeuralNetwork singleLayerNeuralNetwork = new SingleLayerNeuralNetwork(0.5, 0.0, classesAmount);
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics(singleLayerNeuralNetwork, splitDataset);
        System.out.println("Testing of the Single Layer Neural Network algorithm");
        evaluationMetrics.measureAccuracy();
        double precision = evaluationMetrics.evaluatePrecision(splitDataset.getTestSetLabels().getFirst());
        double recall = evaluationMetrics.evaluateRecall(splitDataset.getTestSetLabels().getFirst());
        evaluationMetrics.evaluateF1Measure(precision, recall);
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
                case "1" -> predictFromUserInput(splitDataset, scanner, new KNearestNeighbours(), new Perceptron(0.5), encoder);
                case "2" -> exit = false;
                default -> System.out.println("Unknown action. Try again.");
            }
        }

        scanner.close();
    }

    private static void predictFromUserInput(SplitDataset splitDataset, Scanner scanner,
                                             KNearestNeighbours knnClassifier,
                                             Perceptron perceptronClassifier,
                                             LabelEncoder encoder) {
        System.out.println("Choose a classifier:\n1) KNN\n2) Perceptron");
        String choice = scanner.nextLine();

        Classifier chosenClassifier;
        boolean isPerceptron = false;

        if ("2".equals(choice)) {
            chosenClassifier = perceptronClassifier;
            isPerceptron = true;
        } else {
            chosenClassifier = knnClassifier;
        }

        chosenClassifier.train(splitDataset.getTrainSet());

        int nearestObservations;
        if (chosenClassifier instanceof KNearestNeighbours) {
            System.out.println("Enter number of nearest observations:");
            boolean validNumber = false;
            while (!validNumber) {
                String nearestObsInput = scanner.nextLine();
                try {
                    nearestObservations = Integer.parseInt(nearestObsInput);
                    validNumber = true;
                    knnClassifier.setK(nearestObservations);
                } catch (NumberFormatException e) {
                    System.err.println("Not a number. Try again.");
                }
            }
        }

        List<Double> vector = new ArrayList<>();
        System.out.println("Enter numbers for the vector (empty line to finish):");
        String input;
        while (!(input = scanner.nextLine()).isEmpty()) {
            try {
                vector.add(Double.parseDouble(input));
            } catch (NumberFormatException e) {
                System.err.println("Not a number. Try again.");
            }
        }

        double[] featureArray = new double[vector.size()];
        for (int i = 0; i < vector.size(); i++) {
            featureArray[i] = vector.get(i);
        }

        int predictedLabel = chosenClassifier.predict(featureArray);
        System.out.println("Predicted label (encoded integer): " + encoder.decode(predictedLabel));

        if (isPerceptron) {
            double minX = 0.0;
            double maxX = 10.0;
            int steps = 50;

            Perceptron p = (Perceptron) chosenClassifier;
            DecisionBoundaryPlotter plotter = new DecisionBoundaryPlotter(p, minX, maxX, steps);

            List<Pair<Integer, double[]>> someTestPoints = new ArrayList<>();

            System.out.println("ASCII plot of the decision boundary:");
            plotter.asciiPlot(someTestPoints);

            plotter.exportBoundaryToCsv("src/main/resources/boundary.csv");
        }
    }
}