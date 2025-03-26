package org.knn.evaluation;

import org.knn.data.SplitDataset;
import org.knn.models.Classifier;

import java.util.List;

public class EvaluationMetrics {
    Classifier classifier;
    SplitDataset dataset;

    public EvaluationMetrics(Classifier classifier, SplitDataset dataset) {
        this.classifier = classifier;
        this.dataset = dataset;
        this.classifier.train(dataset.getTrainSet());
    }

    public double measureAccuracy() {

        int correctPredictionsCount = 0;

        List<double[]> testSetVectors = dataset.getTestSetVectors();
        List<Integer> testSetLabels = dataset.getTestSetLabels();

        for (int i = 0; i < testSetVectors.size(); i++) {
            int answer = classifier.predict(testSetVectors.get(i));
            if (answer == testSetLabels.get(i)) correctPredictionsCount++;
        }

        double number = (double) correctPredictionsCount / testSetVectors.size();
        System.out.printf("Evaluated accuracy is: %.2f%%%n", number * 100);
        return number;
    }

    public double evaluatePrecision(int positiveClass) {
        int truePositives = 0;
        int falsePositives = 0;

        List<double[]> testSetVectors = dataset.getTestSetVectors();
        List<Integer> testSetLabels = dataset.getTestSetLabels();

        for (int i = 0; i < testSetVectors.size(); i++) {

            int answer = classifier.predict(testSetVectors.get(i));
            int correctAnswer = testSetLabels.get(i);

            if (answer == correctAnswer && positiveClass == correctAnswer) truePositives++;
            else if (answer != correctAnswer && positiveClass == correctAnswer) falsePositives++;
        }

        double number = (double) truePositives / (truePositives + falsePositives);
        System.out.printf("Evaluated precision is: %.2f%%%n", number * 100);
        return number;
    }

    public double evaluateRecall(int positiveClass) {
        int truePositives = 0;
        int falseNegatives = 0;

        List<double[]> testSetVectors = dataset.getTestSetVectors();
        List<Integer> testSetLabels = dataset.getTestSetLabels();


        for (int i = 0; i < testSetVectors.size(); i++) {

            int answer = classifier.predict(testSetVectors.get(i));
            int correctAnswer = testSetLabels.get(i);

            if (answer == correctAnswer && positiveClass == correctAnswer) truePositives++;
            else if (answer != correctAnswer && positiveClass != correctAnswer) falseNegatives++;
        }

        double number = (double) truePositives / (truePositives + falseNegatives);
        System.out.printf("Evaluated recall is: %.2f%%%n", number * 100);
        return number;
    }

    public double evaluateF1Measure(double precision, double recall) {

        double number = (2 * precision * recall) / (precision + recall);
        System.out.printf("Evaluated FMeasure is: %.2f%%%n", number * 100);
        return number;
    }
}
