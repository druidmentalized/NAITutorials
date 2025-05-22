package org.nai.algorithms;

import org.nai.structures.Pair;
import org.nai.structures.Vector;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Knapsack {
    private final Random random = new Random();

    private final Vector itemWeights;
    private final Vector itemValues;
    private final double capacity;

    public Knapsack(Vector itemWeights, Vector itemValues, double capacity) {
        this.itemWeights = itemWeights;
        this.itemValues = itemValues;
        this.capacity = capacity;
    }

    public Pair<List<Vector>, Double> bruteForce() {
        int n = itemWeights.size();

        List<List<Integer>> bestSubsetsIndices = new ArrayList<>();

        double maxValue = findBestSubsets(bestSubsetsIndices, n);

        List<Vector> bestVectors = new ArrayList<>(bestSubsetsIndices.size());
        for (List<Integer> indices : bestSubsetsIndices) {
            double[] pickedWeights = new double[indices.size()];
            for (int j = 0; j < indices.size(); j++) {
                pickedWeights[j] = itemWeights.get(indices.get(j));
            }
            bestVectors.add(new Vector(pickedWeights));
        }

        return new Pair<>(bestVectors, maxValue);
    }


    private double findBestSubsets(List<List<Integer>> bestSubsetsIndices, int n) {
        double maxValue = Double.NEGATIVE_INFINITY;
        int totaBitMasks = 1 << n;

        for (int mask = 0; mask < totaBitMasks; mask++) {
            double sumWeights = 0.0;
            double sumValues = 0.0;
            List<Integer> indices = new ArrayList<>();

            for (int i = 0; i < n; i++) {
                if ((mask & (1 << i)) != 0) {
                    indices.add(i);
                    sumWeights += itemWeights.get(i);
                    sumValues += itemValues.get(i);
                }
            }

            if (sumWeights <= capacity) {
                if (sumValues > maxValue) {
                    bestSubsetsIndices.clear();
                    bestSubsetsIndices.add(indices);
                    maxValue = sumValues;
                } else if (sumValues == maxValue) {
                    bestSubsetsIndices.add(indices);
                }
            }
        }

        return maxValue;
    }

    public Pair<Vector, Double> greedyDensityApproach() {
        int n = itemWeights.size();

        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (i, j) ->
                Double.compare(
                        itemValues.get(j) / itemWeights.get(j),
                        itemValues.get(i) / itemWeights.get(i)
                )
        );

        double[] picked = new double[n];
        double weightSum = 0;
        double valueSum = 0;
        int arrIdx = 0;

        for (int itemIdx : idx) {
            double weight = itemWeights.get(itemIdx);
            double value = itemValues.get(itemIdx);

            if (weightSum + weight <= capacity) {
                picked[arrIdx++] = weight;
                weightSum += weight;
                valueSum += value;
            }
        }

        return new Pair<>(
                new Vector(Arrays.copyOf(picked, arrIdx)),
                valueSum
        );
    }

    public Pair<Vector, Double> hillClimbing(int numberOfRestarts) {
        Vector bestWeights = null;
        double maxValue = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < numberOfRestarts; i++) {
            Set<Integer> available = initializeIndices();
            int start = pickRandomStart(available);
            List<Integer> picked = localSearch(available, start);
            double totalValue = computeTotalValue(picked);

            if (totalValue > maxValue) {
                maxValue = totalValue;
                bestWeights = extractWeights(picked);
            }
        }

        return new Pair<>(bestWeights, maxValue);
    }

    private Set<Integer> initializeIndices() {
        return IntStream.range(0, itemWeights.size()).boxed().collect(Collectors.toSet());
    }

    private int pickRandomStart(Set<Integer> available) {
        List<Integer> list = new ArrayList<>(available);
        int start = list.get(random.nextInt(list.size()));
        available.remove(start);
        return start;
    }

    private List<Integer> localSearch(Set<Integer> available, int start) {
        List<Integer> selected = new ArrayList<>(List.of(start));
        double currentWeight = itemWeights.get(start);
        double currentValue = itemValues.get(start);

        boolean improved = true;
        while (improved) {
            improved = false;
            int bestIdx = -1;
            double bestVal = currentValue;

            for (int i : available) {
                double newWeight = currentWeight + itemWeights.get(i);
                double newVal = currentValue + itemValues.get(i);

                if (newWeight > capacity || newVal <= bestVal) continue;

                bestIdx = i;
                bestVal = newVal;
                improved = true;
            }

            if (!improved) continue;
            available.remove(bestIdx);
            selected.add(bestIdx);
            currentWeight += itemWeights.get(bestIdx);
            currentValue = bestVal;
        }

        return selected;
    }

    private double computeTotalValue(List<Integer> indices) {
        return indices.stream().mapToDouble(itemValues::get).sum();
    }

    private Vector extractWeights(List<Integer> indices) {
        return new Vector(indices.stream().mapToDouble(itemWeights::get).toArray());
    }
}
