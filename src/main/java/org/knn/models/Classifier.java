package org.knn.models;

import org.knn.structures.Pair;

import java.util.List;

public interface Classifier {
    void train(List<Pair<Integer, double[]>> trainSet);
    int predict(double[] input);
}
