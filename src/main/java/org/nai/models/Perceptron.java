package org.nai.models;

import org.nai.data.Dataset;
import org.nai.structures.Vector;

public class Perceptron implements Classifier {
    private Vector weights;
    private double alpha;

    private int epochs;
    private double threshold = 0;

    public Perceptron(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public void train(Dataset trainSet) {
        weights = new Vector(new double[trainSet.getData().getFirst().second().size()]);
        boolean shouldBreak = false;
        for (; ; epochs++) {
            int errors = (int) trainSet.getData().stream().filter(pair -> adjustWeights(pair.second(), pair.first())).count();

            if (errors == 0) {
                System.out.println("Training complete after " + (epochs + 1) + " epochs.\n");
                shouldBreak = true;
            }
            else if (epochs >= 1000) {
                System.out.println("Training forcibly stopped after " + (epochs) + " epochs.\n");
                shouldBreak = true;
            }

            if (shouldBreak) break;
        }
    }

    private boolean adjustWeights(Vector vector, double answer) {
        int prediction = predict(vector);
        double delta = answer - prediction;

        if (delta != 0) {
            for (int i = 0; i < weights.size(); i++) {
                weights = weights.add(vector.scale(alpha * delta));
            }

            threshold -= delta * alpha;
            return true;
        }

        return false;
    }

    @Override
    public int predict(Vector vector) {
        return (vector.dot(weights) >= threshold) ? 1 : 0;
    }

    // Helper

    public double netValue(Vector vector) {
        return vector.dot(weights) - threshold;
    }

    // Getters & Setters

    public Vector getWeights() {
        return weights;
    }
    public double getThreshold() {
        return threshold;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }
}
