package org.knn.knn;

import org.knn.models.Pair;

import java.util.*;

public class KNearestNeighbours {
    private final int k;
    private final List<Pair<String, List<Double>>> trainSet;
    private final List<Pair<String, Double>> distances = new ArrayList<>();

    public KNearestNeighbours(int k, List<Pair<String, List<Double>>> trainSet) {
        this.k = k;
        this.trainSet = trainSet;
    }

    public String run(List<Double> vector) {
        calculateDistances(vector);
        sortDistances();
        return predict();
    }

    private void calculateDistances(List<Double> vector) {
        distances.clear();
        for (Pair<String, List<Double>> pair : trainSet) {
            distances.add(new Pair<>(pair.first(), calculateEuclideanDistance(pair.second(), vector)));
        }
    }

    private double calculateEuclideanDistance(List<Double> vec1, List<Double> vec2) {
        double result = 0;

        if (vec1.size() != vec2.size()) {
            throw new IllegalArgumentException("Vectors must be the same dimension");
        }

        for (int i = 0; i < vec1.size(); i++) {
            result += Math.pow((vec1.get(i) - vec2.get(i)), 2);
        }

        result = Math.sqrt(result);

        return result;
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

    private String findPredictedClass() {
        HashMap<String, Integer> countMap = new HashMap<>();
        int maxCount = 0;
        List<String> mostFrequentClasses = new ArrayList<>();

        for (int i = 0; i < this.k; i++) {
            Pair<String, Double> pair = distances.get(i);
            String label = pair.first();
            int count = countMap.getOrDefault(label, 0) + 1;
            countMap.put(label, count);

            if (count > maxCount) {
                maxCount = count;
                mostFrequentClasses.clear();
                mostFrequentClasses.add(label);
            } else if (count == maxCount) {
                mostFrequentClasses.add(label);
            }
        }

        if (mostFrequentClasses.size() > 1) {
            return mostFrequentClasses.get(new Random().nextInt(mostFrequentClasses.size()));
        }

        return mostFrequentClasses.getFirst();
    }

    private String predict() {
        return findPredictedClass();
    }
}