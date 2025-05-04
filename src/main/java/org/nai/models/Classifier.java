package org.nai.models;

import org.nai.structures.Pair;

import java.util.List;
import java.util.Map;

public interface Classifier extends Model {
    void train(List<Pair<Integer, double[]>> trainSet);
    int predict(double[] input);
}
