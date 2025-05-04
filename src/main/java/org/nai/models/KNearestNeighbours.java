package org.nai.models;

import org.nai.structures.Pair;
import org.nai.utils.DistanceUtils;

import java.util.*;

public class KNearestNeighbours implements Classifier {
    private List<Pair<Integer, double[]>> trainSet;
    private int k;
    private List<Pair<Integer, Double>> distances;

    public KNearestNeighbours() {
        k = 3;
    }

    public KNearestNeighbours(int k) {
        this.k = k;
    }

    private List<Pair<Integer, Double>> calculateDistances(double[] vector) {
        List<Pair<Integer, Double>> distances = new ArrayList<>();
        for (Pair<Integer, double[]> pair : trainSet) {
            distances.add(new Pair<>(pair.first(), DistanceUtils.calculateEuclideanDistance(pair.second(), vector)));
        }
        return distances;
    }

    private void sortDistances() {
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
        int i = 0, j = 0, y = 0;

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

    private int findPredictedClass() {
        HashMap<Integer, Integer> countMap = new HashMap<>();
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
            return mostFrequentClasses.get(new Random().nextInt(mostFrequentClasses.size()));
        }

        return mostFrequentClasses.getFirst();
    }

    @Override
    public void train(List<Pair<Integer, double[]>> trainSet) {
        this.trainSet = trainSet;
    }

    @Override
    public int predict(double[] vector) {
        distances = calculateDistances(vector);
        sortDistances();
        return findPredictedClass();
    }

    // Helper

    public void setK(int k) {
        this.k = k;
    }
}