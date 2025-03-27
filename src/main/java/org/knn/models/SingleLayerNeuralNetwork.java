package org.knn.models;

import org.knn.structures.Pair;

import java.util.ArrayList;
import java.util.List;

public class SingleLayerNeuralNetwork implements Classifier {

    private final List<Perceptron> neurons = new ArrayList<>();
    private final double alpha;
    private final double beta;

    public SingleLayerNeuralNetwork(double alpha, double beta, int classes) {
        this.alpha = alpha;
        this.beta = beta;
        for (int i = 0; i < classes; i++) {
            neurons.add(new Perceptron(alpha));
        }
    }

    @Override
    public void train(List<Pair<Integer, double[]>> trainSet) {
        for (int classIndex = 0; classIndex < neurons.size(); classIndex++) {
            List<Pair<Integer, double[]>> binaryTrainSet = new ArrayList<>();

            for (Pair<Integer, double[]> sample : trainSet) {
                int label = sample.first().equals(classIndex) ? 1 : 0;
                binaryTrainSet.add(new Pair<>(label, sample.second()));
            }

            neurons.get(classIndex).train(binaryTrainSet);
        }
    }

    @Override
    public int predict(double[] input) {
        int bestClass = -1;
        double highestNet = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < neurons.size(); i++) {
            Perceptron neuron = neurons.get(i);
            double netValue = neuron.dotProduct(input, neuron.getWeights()) - neuron.getThreshold();
            int prediction = neuron.predict(input);

            if (prediction == 1 && netValue > highestNet) {
                highestNet = netValue;
                bestClass = i;
            }
        }

        return bestClass;
    }
}
