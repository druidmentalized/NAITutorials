package org.nai.evaluation;

import org.nai.data.SplitDataset;
import org.nai.models.Classifier;
import org.nai.models.Clusterer;
import org.nai.structures.Cluster;
import org.nai.structures.Vector;

import java.util.List;

public class EvaluationMetrics {
    private final Classifier classifier;
    private final SplitDataset dataset;

    private final Clusterer clusterer;
    private final List<Vector> clusteringData;
    private final Integer clusteringK;

    public EvaluationMetrics(Classifier classifier, SplitDataset dataset) {
        this.classifier = classifier;
        this.dataset = dataset;
        this.clusterer = null;
        this.clusteringData = null;
        this.clusteringK = null;
        this.classifier.train(dataset.trainSet());
    }

    public EvaluationMetrics(Clusterer clusterer, List<Vector> data, int k) {
        this.classifier = null;
        this.dataset = null;
        this.clusterer = clusterer;
        this.clusteringData = data;
        this.clusteringK = k;
    }

    public double measureAccuracy() {
        checkClassifier();
        int correctPredictionsCount = 0;

        List<Vector> testSetVectors = dataset.testSet().getVectors();
        List<Integer> testSetLabels = dataset.testSet().getLabels();

        for (int i = 0; i < testSetVectors.size(); i++) {
            int answer = classifier.predict(testSetVectors.get(i));
            if (answer == testSetLabels.get(i)) correctPredictionsCount++;
        }

        double number = (double) correctPredictionsCount / testSetVectors.size();
        System.out.printf("Evaluated accuracy is: %.2f%%%n", number * 100);
        return number;
    }

    public double evaluatePrecision(int positiveClass) {
        checkClassifier();
        int truePositives = 0;
        int falsePositives = 0;

        List<Vector> testSetVectors = dataset.testSet().getVectors();
        List<Integer> testSetLabels = dataset.testSet().getLabels();

        for (int i = 0; i < testSetVectors.size(); i++) {

            int answer = classifier.predict(testSetVectors.get(i));
            int correctAnswer = testSetLabels.get(i);

            if (answer == correctAnswer && positiveClass == correctAnswer) truePositives++;
            else if (answer != correctAnswer && positiveClass == correctAnswer) falsePositives++;
        }

        int trueFalsePositives = truePositives + falsePositives;
        double number = 0;
        if (trueFalsePositives != 0) {
            number = (double) truePositives / trueFalsePositives;
        }
        System.out.printf("Evaluated precision is: %.2f%%%n", number * 100);
        return number;
    }

    public double evaluateRecall(int positiveClass) {
        checkClassifier();
        int truePositives = 0;
        int falseNegatives = 0;

        List<Vector> testSetVectors = dataset.testSet().getVectors();
        List<Integer> testSetLabels = dataset.testSet().getLabels();


        for (int i = 0; i < testSetVectors.size(); i++) {

            int answer = classifier.predict(testSetVectors.get(i));
            int correctAnswer = testSetLabels.get(i);

            if (answer == correctAnswer && positiveClass == correctAnswer) truePositives++;
            else if (answer != correctAnswer && positiveClass != correctAnswer) falseNegatives++;
        }

        int truePositivesFalseNegatives = truePositives + falseNegatives;
        double number = 0;
        if (truePositivesFalseNegatives != 0) {
            number = (double) truePositives / truePositivesFalseNegatives;
        }
        System.out.printf("Evaluated recall is: %.2f%%%n", number * 100);
        return number;
    }

    public double evaluateF1Measure(double precision, double recall) {
        checkClassifier();
        double number = (2 * precision * recall) / (precision + recall);
        System.out.printf("Evaluated FMeasure is: %.2f%%%n", number * 100);
        return number;
    }

    public double evaluateWCSS() {
        checkClusterer();
        List<Cluster> clusters = clusterer.groupClusters(clusteringK, clusteringData);
        double wcss = computeWCSS(clusters);
        System.out.printf("Computed WCSS: %.4f%n", wcss);
        return wcss;
    }

    public static double computeWCSS(List<Cluster> clusters) {
        double wcss = 0;
        for (Cluster cluster : clusters) {
            Vector center = cluster.getCentroid().coordinates();
            for (Vector vector : cluster.getMembers()) {
                wcss += center.squaredDistanceTo(vector);
            }
        }
        return wcss;
    }

    // Helper

    private void checkClassifier() {
        if (classifier == null) {
            throw new IllegalStateException("No Classifier provided");
        }
    }

    private void checkClusterer() {
        if (clusterer == null) {
            throw new IllegalStateException("No Clusterer provided");
        }
    }
}
