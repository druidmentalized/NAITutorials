package org.nai.models;

import org.nai.data.Dataset;
import org.nai.structures.Pair;
import org.nai.structures.Vector;

import java.util.List;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Map;
import java.util.HashMap;
import java.util.Random;

public class KNearestNeighbours implements Classifier {
    private final Random random = new Random();

    private Dataset trainSet;
    private int k;

    public KNearestNeighbours() {
        k = 3;
    }

    private List<Pair<Integer, Double>> calculateDistances(Vector vector) {
        List<Pair<Integer, Double>> distances = new ArrayList<>();
        for (Pair<Integer, Vector> pair : trainSet.getData()) {
            distances.add(new Pair<>(pair.first(), vector.distanceTo(pair.second())));
        }
        return distances;
    }

    private void sortDistances(List<Pair<Integer, Double>> distances) {
        mergeSort(distances);
    }

    public static <K, V extends Comparable<V>> void mergeSort(List<Pair<K, V>> list) {
        if (list.size() < 2) return;

        int mid = list.size() / 2;
        List<Pair<K, V>> left = new LinkedList<>(list.subList(0, mid));
        List<Pair<K, V>> right = new LinkedList<>(list.subList(mid, list.size()));

        mergeSort(left);
        mergeSort(right);
        merge(list, left, right);
    }

    private static <K, V extends Comparable<V>> void merge(List<Pair<K, V>> list, List<Pair<K, V>> left, List<Pair<K, V>> right) {
        int i = 0;
        int j = 0;
        int y = 0;

        while (i < left.size() && j < right.size()) {
            if (left.get(i).second().compareTo(right.get(j).second()) <= 0) {
                list.set(y++, left.get(i++));
            } else {
                list.set(y++, right.get(j++));
            }
        }
        while (i < left.size()) {
            list.set(y++, left.get(i++));
        }
        while (j < right.size()) {
            list.set(y++, right.get(j++));
        }
    }

    private int findPredictedClass(List<Pair<Integer, Double>> distances) {
        Map<Integer, Integer> countMap = new HashMap<>();
        int maxCount = 0;
        List<Integer> mostFrequentClasses = new ArrayList<>();

        for (int i = 0; i < this.k; i++) {
            Pair<Integer, Double> pair = distances.get(i);
            int labelEncoded = pair.first();
            int count = countMap.getOrDefault(labelEncoded, 0) + 1;
            countMap.put(labelEncoded, count);

            if (count > maxCount) {
                maxCount = count;
                mostFrequentClasses.clear();
                mostFrequentClasses.add(labelEncoded);
            } else if (count == maxCount) {
                mostFrequentClasses.add(labelEncoded);
            }
        }

        if (mostFrequentClasses.size() > 1) {
            return mostFrequentClasses.get(random.nextInt(mostFrequentClasses.size()));
        }

        return mostFrequentClasses.getFirst();
    }

    @Override
    public void train(Dataset trainSet) {
        this.trainSet = trainSet;
    }

    @Override
    public int predict(Vector vector) {
        var distances = calculateDistances(vector);
        sortDistances(distances);
        return findPredictedClass(distances);
    }

    // Helper

    public void setK(int k) {
        this.k = k;
    }
}