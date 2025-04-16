package org.nai.models;

import org.nai.structures.Pair;

import java.util.List;

public interface Classifier {
    void train(List<Pair<Integer, double[]>> trainSet);
    int predict(double[] input);
}
