package org.knn.models;

import org.knn.structures.Pair;

import java.util.Arrays;
import java.util.List;

public class Perceptron implements Classifier {
    private double[] weights;
    private final double alpha;

    private int epochs;
    private double threshold = 0;

    public Perceptron(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public void train(List<Pair<Integer, double[]>> trainSet) {
        System.out.println("Training of the perceptron:");

        weights = new double[trainSet.getFirst().second().length];
        for (; ; epochs++) {
            int errors = 0;
            for (Pair<Integer, double[]> pair : trainSet) {
                if (adjustWeights(pair.second(), pair.first())) errors++;
            }

            System.out.println("Epoch " + (epochs + 1) + ": Weights = " + Arrays.toString(weights) + ", Threshold = " + threshold);
            double number = (double) (trainSet.size() - errors) / trainSet.size();
            System.out.printf("Correct predictions count is: %.2f%%%n\n", number * 100);

            if (errors == 0) {
                System.out.println("Training complete after " + (epochs + 1) + " epochs.\n");
                break;
            }
            else if (epochs > 100) {
                System.out.println("Training forcibly stopped after " + (epochs + 1) + " epochs.\n");
                break;
            }
        }
    }

    private boolean adjustWeights(double[] vector, double answer) {
        int prediction = predict(vector);
        double delta = answer - prediction;

        if (delta != 0) {
            for (int i = 0; i < weights.length; i++) {
                weights[i] += alpha * delta * vector[i];
            }

            threshold -= delta * alpha;
            return true;
        }

        return false;
    }

    @Override
    public int predict(double[] vector) {
        return (dotProduct(vector, weights) >= threshold) ? 1 : 0;
    }

    // Helper
    public double dotProduct(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            System.err.println("Can't calculate dot product of different vectors!");
            return 0;
        }

        double sum = 0;

        for (int i = 0; i < vec1.length; i++) {
            sum += vec1[i] * vec2[i];
        }

        return sum;
    }

    // Getters & Setters


    public double[] getWeights() {
        return weights;
    }
    public double getThreshold() {
        return threshold;
    }
}
