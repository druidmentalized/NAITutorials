package org.knn.main;

import org.knn.data.PrepareDataset;
import org.knn.data.SplitDataset;
import org.knn.evaluation.EvaluationMetrics;
import org.knn.models.Classifier;
import org.knn.models.KNearestNeighbours;
import org.knn.models.Perceptron;
import org.knn.models.SingleLayerNeuralNetwork;
import org.knn.plot.DecisionBoundaryPlotter;
import org.knn.utils.LabelEncoder;

import java.util.*;

public class Main {
    public static void main(String[] args) {
        LabelEncoder encoder = new LabelEncoder();
        PrepareDataset prepareDataset = new PrepareDataset();

        //var dataset = prepareDataset.parseDataset("src/main/resources/languagesdataset", encoder, false);
        //SplitDataset splitDataset = prepareDataset.trainTestSplit(dataset, 0.66);

        var trainSet = prepareDataset.parseDataset("src/main/resources/csv/lang.train.csv", encoder, true);
        var testSet = prepareDataset.parseDataset("src/main/resources/csv/lang.test.csv", encoder, true);
        SplitDataset splitDataset = new SplitDataset(trainSet, testSet);

        int classesAmount = encoder.getClassesAmount();
        runKNNTests(splitDataset);
        System.out.println();
        runPerceptronTests(splitDataset, classesAmount);
        System.out.println();
        runSingleLayerNeuralNetworkTests(splitDataset, classesAmount);
        System.out.println();


        startUserInput(splitDataset, encoder);
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

    private static void runPerceptronTests(SplitDataset splitDataset, int classesAmount) {
        Perceptron perceptron = new Perceptron(0.2);
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics(perceptron, splitDataset);
        System.out.println("Testing of the Perceptron algorithm");
        outputEvaluations(evaluationMetrics, classesAmount);
    }

    private static void runSingleLayerNeuralNetworkTests(SplitDataset splitDataset, int classesAmount) {
        SingleLayerNeuralNetwork singleLayerNeuralNetwork = new SingleLayerNeuralNetwork(0.2, 0.0, classesAmount);
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics(singleLayerNeuralNetwork, splitDataset);
        System.out.println("Testing of the Single Layer Neural Network algorithm");
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

    private static void predictFromUserInput(SplitDataset splitDataset, Scanner scanner, LabelEncoder encoder) {

        System.out.println("Choose a classifier:\n1) KNN\n2) Perceptron\n3) Single Layer Neural Network");
        String choice = scanner.nextLine();

        Classifier chosenClassifier;
        boolean isPerceptron = false;

        System.out.println("Enter learning rate (alpha):");
        double alpha = 0.1;
        boolean validAlpha = false;
        while (!validAlpha) {
            try {
                alpha = Double.parseDouble(scanner.nextLine());
                validAlpha = true;
            } catch (NumberFormatException e) {
                System.err.println("Invalid number. Try again.");
            }
        }

        switch (choice) {
            case "2":
                chosenClassifier = new Perceptron(alpha);
                isPerceptron = true;
                break;
            case "3":
                int classesAmount = encoder.getClassesAmount();
                chosenClassifier = new SingleLayerNeuralNetwork(alpha, 0.0, classesAmount);
                break;
            default:
                chosenClassifier = new KNearestNeighbours();
        }


        chosenClassifier.train(splitDataset.getTrainSet());

        if (chosenClassifier instanceof KNearestNeighbours) {
            System.out.println("Enter number of nearest observations:");
            boolean validNumber = false;
            while (!validNumber) {
                String nearestObsInput = scanner.nextLine();
                try {
                    int nearestObservations = Integer.parseInt(nearestObsInput);
                    ((KNearestNeighbours) chosenClassifier).setK(nearestObservations);
                    validNumber = true;
                } catch (NumberFormatException e) {
                    System.err.println("Not a number. Try again.");
                }
            }
        }

        System.out.println("Enter '1' for numeric vector input or '2' for text input:");
        String inputType = scanner.nextLine();

        double[] featureArray;

        if ("2".equals(inputType)) {
            System.out.println("Enter your text:");
            String textInput = scanner.nextLine();
            featureArray = PrepareDataset.textToVector(textInput);
        } else {
            System.out.println("Enter numbers for the vector (comma-separated):");
            String numericLine = scanner.nextLine();
            String[] tokens = numericLine.trim().split(",");
            featureArray = new double[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                try {
                    featureArray[i] = Double.parseDouble(tokens[i].trim());
                } catch (NumberFormatException e) {
                    System.err.println("Invalid number '" + tokens[i] + "'. Defaulting to 0.");
                    featureArray[i] = 0;
                }
            }
        }

        int predictedLabel = chosenClassifier.predict(featureArray);
        System.out.println("Predicted label without encoding: " + predictedLabel);
        System.out.println("Predicted label: " + encoder.decode(predictedLabel));

        if (isPerceptron) {
            double minX = 0.0;
            double maxX = 10.0;
            int steps = 50;

            Perceptron p = (Perceptron) chosenClassifier;
            DecisionBoundaryPlotter plotter = new DecisionBoundaryPlotter(p, minX, maxX, steps);

            System.out.println("ASCII plot of the decision boundary:");
            plotter.asciiPlot(splitDataset.getTestSet());

            plotter.exportBoundaryToCsv("src/main/resources/boundary.csv");
        }
    }

}