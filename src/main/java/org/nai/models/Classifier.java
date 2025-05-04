package org.nai.models;

import org.nai.data.Dataset;
import org.nai.structures.Vector;

public interface Classifier extends Model {
    void train(Dataset trainSet);
    int predict(Vector input);
}
