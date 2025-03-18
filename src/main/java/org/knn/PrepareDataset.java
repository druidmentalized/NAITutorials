package org.knn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class PrepareDataset {
    private final List<Pair<String, List<Double>>> trainSet = new ArrayList<>();
    private final List<Pair<String, List<Double>>> testSet = new ArrayList<>();


    public List<Pair<String, List<Double>>> parseDataset(String filePath) {
        List<Pair<String, List<Double>>> dataset = new ArrayList<>();

        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(filePath))) {
            String line;

            while ((line = bufferedReader.readLine()) != null) {
                String[] tokens = line.split(",");
                List<Double> vector = new ArrayList<>();
                String className = "";
                for (String token : tokens) {
                    try {
                        vector.add(Double.parseDouble(token));
                    }
                    catch (NumberFormatException e) {
                        className = token;
                        break;
                    }
                }
                dataset.add(new Pair<>(className, vector));
            }

        } catch (IOException e) {
            System.err.println("Reading data went wrong: " + e.getMessage());
        }

        return dataset;
    }

    public void trainTestSplit(List<Pair<String, List<Double>>> dataset) {
        Map<String, List<Pair<String, List<Double>>>> classToSamples = new HashMap<>();

        for (Pair<String, List<Double>> pair : dataset) {
            classToSamples.computeIfAbsent(pair.getFirst(), _ -> new ArrayList<>()).add(pair);
        }

        for (List<Pair<String, List<Double>>> samples : classToSamples.values()) {
            Collections.shuffle(samples);
        }

        for (String className : classToSamples.keySet()) {
            List<Pair<String, List<Double>>> samples = classToSamples.get(className);
            int totalClassSize = samples.size();
            int trainCount = Math.max(1, (int) Math.round(totalClassSize * 0.66));

            trainSet.addAll(samples.subList(0, trainCount));
            testSet.addAll(samples.subList(trainCount, totalClassSize));
        }

        Collections.shuffle(trainSet);
        Collections.shuffle(testSet);
    }

    public List<Pair<String, List<Double>>> getTrainSet() {
        return trainSet;
    }
    public List<Pair<String, List<Double>>> getTestSet() {
        return testSet;
    }
}
