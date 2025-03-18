package org.knn;

import java.util.*;

public class Main {
    private final static List<Pair<String, List<Double>>> trainSet = new ArrayList<>();

    public static void main(String[] args) {
        PrepareDataset prepareDataset = new PrepareDataset();
        prepareDataset.trainTestSplit(prepareDataset.parseDataset("src/main/resources/iris.csv"));

        /*prepareTrainSet();

        List<Double> vector1 = new ArrayList<>() {{
            add(4.0);
            add(4.0);
            add(0.0);
        }};

        List<Double> vector2 = new ArrayList<>() {{
            add(1.0);
            add(1.0);
            add(5.0);
        }};

        List<Double> vector3 = new ArrayList<>() {{
            add(6.0);
            add(0.0);
            add(0.0);
        }};

        runKNN(vector1);
        runKNN(vector2);
        runKNN(vector3);*/
    }

    private static void prepareTrainSet() {
        trainSet.add(new Pair<>("A", Arrays.asList(5.0, 4.0, 1.0)));
        trainSet.add(new Pair<>("A", Arrays.asList(4.0, 3.0, 0.0)));

        trainSet.add(new Pair<>("B", Arrays.asList(1.0, 2.0, 3.0)));
        trainSet.add(new Pair<>("B", Arrays.asList(2.0, 0.0, 4.0)));

        trainSet.add(new Pair<>("C", Arrays.asList(6.0, 1.0, 1.0)));
        trainSet.add(new Pair<>("C", Arrays.asList(5.0, 0.0, 1.0)));
    }

    private static void runKNN(List<Double> vector) {
        KNearestNeighbours kNearestNeighbours = new KNearestNeighbours(3, trainSet);
        kNearestNeighbours.run(vector);
    }
}