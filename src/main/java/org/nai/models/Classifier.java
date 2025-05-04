package org.nai.models;

import org.nai.data.Dataset;
import org.nai.structures.Pair;
import org.nai.structures.Vector;

import java.util.List;
import java.util.Map;

public interface Classifier extends Model {
    void train(Dataset trainSet);
    int predict(Vector input);
}
