package org.nai.algorithms;

import org.nai.structures.Pair;
import org.nai.structures.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Knapsack {
    private Vector itemWeights;
    private Vector itemValues;
    private double capacity;

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

    public Pair<Vector,Double> greedyDensityApproach() {
        int n = itemWeights.size();

        Integer[] idx = new Integer[n];
        for(int i=0;i<n;i++) idx[i]=i;
        Arrays.sort(idx, (i,j) ->
                Double.compare(
                        itemValues.get(j)/itemWeights.get(j),
                        itemValues.get(i)/itemWeights.get(i)
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
}
