package org.nai.models;

import org.nai.data.Dataset;
import org.nai.structures.Pair;
import org.nai.structures.Vector;

import java.util.ArrayList;
import java.util.List;

public class SingleLayerNeuralNetwork implements Classifier {

    private final List<Perceptron> neurons = new ArrayList<>();

    public SingleLayerNeuralNetwork(double alpha, int classes) {
        for (int i = 0; i < classes; i++) {
            neurons.add(new Perceptron(alpha));
        }
    }

    @Override
    public void train(Dataset trainSet) {
        for (int classIndex = 0; classIndex < neurons.size(); classIndex++) {
            Dataset binaryTrainSet = new Dataset();

            for (Pair<Integer, Vector> sample : trainSet.getData()) {
                int label = sample.first().equals(classIndex) ? 1 : 0;
                binaryTrainSet.add(new Pair<>(label, sample.second()));
            }

            System.out.println("Training perceptron for class " + classIndex + ":");
            neurons.get(classIndex).train(binaryTrainSet);
        }
    }

    @Override
    public int predict(Vector input) {
        int bestClass = -1;
        double highestNet = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < neurons.size(); i++) {
            Perceptron perceptron = neurons.get(i);
            double netValue = perceptron.netValue(input);

            if (netValue > highestNet) {
                highestNet = netValue;
                bestClass = i;
            }
        }

        return bestClass;
    }
}
