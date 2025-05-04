package org.nai.data;

import org.nai.structures.Pair;
import org.nai.structures.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class SplitDataset {
    private final Dataset trainSet;
    private final Dataset testSet;

    public SplitDataset(Dataset trainSet, Dataset testSet) {
        this.trainSet = trainSet;
        this.testSet = testSet;
    }

    public Dataset getTrainSet() {
        return trainSet;
    }
    public Dataset getTestSet() {
        return testSet;
    }
}
