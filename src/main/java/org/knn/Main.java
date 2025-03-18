package org.knn;

import java.util.*;

public class Main {
    public static void main(String[] args) {
        PrepareDataset prepareDataset = new PrepareDataset();
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
        evaluationMetrics = new EvaluationMetrics(20, prepareDataset);
        evaluationMetrics.measureAccuracy();
    }
}