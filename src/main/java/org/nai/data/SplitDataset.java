package org.nai.data;

import org.nai.structures.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class SplitDataset {
    private final List<Pair<Integer, double[]>> trainSet;
    private final List<Pair<Integer, double[]>> testSet;

    public SplitDataset() {
        trainSet = new ArrayList<>();
        testSet = new ArrayList<>();
    }

    public SplitDataset(List<Pair<Integer, double[]>> trainSet, List<Pair<Integer, double[]>> testSet) {
        this.trainSet = trainSet;
        this.testSet = testSet;
    }

    public List<Pair<Integer, double[]>> getTrainSet() {
        return trainSet;
    }
    public List<Pair<Integer, double[]>> getTestSet() {
        return testSet;
    }

    public List<double[]> getTestSetVectors() {
        return testSet.stream().map(Pair::second).collect(Collectors.toList());
    }

    public List<Integer> getTestSetLabels() {
        return testSet.stream().map(Pair::first).collect(Collectors.toList());
    }
}
