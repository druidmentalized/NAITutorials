package org.knn;

import org.knn.data.PrepareDataset;
import org.knn.evaluation.EvaluationMetrics;
import org.knn.knn.KNearestNeighbours;

import java.util.*;

public class Main {
    private static final PrepareDataset prepareDataset = new PrepareDataset();

    public static void main(String[] args) {
        prepareDataset.trainTestSplit(prepareDataset.parseDataset("src/main/resources/iris.csv"));
        System.out.println("Testcase 1:");
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics(3, prepareDataset);
        evaluationMetrics.measureAccuracy();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");

        System.out.println("Testcase 2:");
        evaluationMetrics = new EvaluationMetrics(7, prepareDataset);
        evaluationMetrics.measureAccuracy();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");

        System.out.println("Testcase 3:");
        evaluationMetrics = new EvaluationMetrics(11, prepareDataset);
        evaluationMetrics.measureAccuracy();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");

        System.out.println("Testcase 4:");
        evaluationMetrics = new EvaluationMetrics(15, prepareDataset);
        evaluationMetrics.measureAccuracy();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");

        System.out.println("Testcase 5:");
        evaluationMetrics = new EvaluationMetrics(20, prepareDataset);
        evaluationMetrics.measureAccuracy();
        System.out.println("────────────────────────────────────────────────────────────────────────────────────");

        System.out.println();
        startUserInput();
    }

    private static void startUserInput() {
        Scanner scanner = new Scanner(System.in);
        boolean exit = true;
        while (exit) {
            System.out.println("Possible actions:");
            System.out.println("1. Evaluate new vector");
            System.out.println("2. Exit");
            System.out.println("Enter your choice: ");

            switch (scanner.nextLine()) {
                case "1" -> predictFromUserInput(scanner);
                case "2" -> exit = false;
                default -> System.out.println("Unknown action. Try again.");
            }
        }

        scanner.close();
    }

    private static void predictFromUserInput(Scanner scanner) {
        List<Double> vector = new ArrayList<>();
        int nearestObservations = 1;
        String input;

        System.out.println("Enter numbers numbers of the vector(empty line to end input):");
        while (!(input = scanner.nextLine()).isEmpty()) {
            try {
                vector.add(Double.parseDouble(input));
            }
            catch (NumberFormatException e) {
                System.err.println("Not a number. Try again.");
            }
        }

        System.out.println("Enter number of nearest observations:");
        boolean normalNumber = false;
        while (!normalNumber) {
            try {
                nearestObservations = Integer.parseInt(scanner.nextLine());
                normalNumber = true;
            }
            catch (NumberFormatException e) {
                System.err.println("Not a number. Try again.");
            }
        }

        KNearestNeighbours knn = new KNearestNeighbours(nearestObservations, prepareDataset.getTrainSet());
        System.out.println("Your vector with " + nearestObservations + " observations should be in " + knn.run(vector));
    }
}