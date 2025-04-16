package org.nai.models;

import org.nai.structures.Pair;

import java.util.List;

public class NaiveBayesClassifier implements Classifier {

    public NaiveBayesClassifier() {

    }

    @Override
    public void train(List<Pair<Integer, double[]>> trainSet) {

    }

    @Override
    public int predict(double[] input) {
        return 0;
    }
}
