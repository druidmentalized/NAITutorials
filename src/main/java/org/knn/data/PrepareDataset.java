package org.knn.data;

import org.knn.structures.Pair;
import org.knn.utils.LabelEncoder;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class PrepareDataset {
    public List<Pair<Integer, double[]>> parseDataset(String filePath, LabelEncoder encoder) {
        List<Pair<Integer, double[]>> dataset = new ArrayList<>();

        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(filePath))) {
            String line;

            while ((line = bufferedReader.readLine()) != null) {
                String[] tokens = line.split(",");
                List<Double> vectorList = new ArrayList<>();
                String labelStr = "";

                for (String token : tokens) {
                    try {
                        vectorList.add(Double.parseDouble(token));
                    } catch (NumberFormatException e) {
                        labelStr = token;
                        break;
                    }
                }

                int encodedLabel = encoder.encode(labelStr);

                double[] vector = new double[vectorList.size()];
                for (int i = 0; i < vectorList.size(); i++) {
                    vector[i] = vectorList.get(i);
                }

                dataset.add(new Pair<>(encodedLabel, vector));
            }

        } catch (IOException e) {
            System.err.println("Reading data went wrong: " + e.getMessage());
        }

        return dataset;
    }

    public SplitDataset trainTestSplit(List<Pair<Integer, double[]>> dataset, double trainRatio) {
        List<Pair<Integer, double[]>> trainSet = new ArrayList<>();
        List<Pair<Integer, double[]>> testSet = new ArrayList<>();

        Map<Integer, List<Pair<Integer, double[]>>> classToSamples = new HashMap<>();
        for (Pair<Integer, double[]> pair : dataset) {
            classToSamples.computeIfAbsent(pair.first(), _ -> new ArrayList<>()).add(pair);
        }

        for (List<Pair<Integer, double[]>> samples : classToSamples.values()) {
            Collections.shuffle(samples);
        }

        for (List<Pair<Integer, double[]>> samples : classToSamples.values()) {
            Collections.shuffle(samples);
            int totalClassSize = samples.size();
            int trainCount = Math.max(1, (int) Math.round(totalClassSize * trainRatio));

            trainSet.addAll(samples.subList(0, trainCount));
            testSet.addAll(samples.subList(trainCount, totalClassSize));
        }

        return new SplitDataset(trainSet, testSet);
    }
}
