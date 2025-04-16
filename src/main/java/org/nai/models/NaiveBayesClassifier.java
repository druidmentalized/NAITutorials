package org.nai.models;

import org.nai.structures.Pair;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class NaiveBayesClassifier implements Classifier {

    private final int classesAmount;
    private final boolean applySmoothingAll;
    Map<String, Double> cache = new HashMap<>();

    public NaiveBayesClassifier(int classesAmount, boolean applySmoothingAll) {
        this.classesAmount = classesAmount;
        this.applySmoothingAll = applySmoothingAll;
    }

    @Override
    public void train(List<Pair<Integer, double[]>> trainSet) {
        cache.clear();

        int entriesAmount = trainSet.size();
        for (int classIndex = 0; classIndex < classesAmount; classIndex++) {
            String className = "Class=" + classIndex;
            int tempClassIndex = classIndex;
            long classQuantity = trainSet.stream().filter(pair -> pair.first() == tempClassIndex).count();

            // Finding priori probabilities
            {
                double classPriorProbability = (double) classQuantity / entriesAmount;
                cache.put(className, classPriorProbability);
            }

            // Finding posteriori probabilities
            {
                int columnsAmount = trainSet.getFirst().second().length;
                for (int columnIndex = 0; columnIndex < columnsAmount; columnIndex++) {
                    int tempColumnIndex = columnIndex;
                    Set<Double> distinctValues = trainSet.stream()
                            .filter(pair -> pair.first() == tempClassIndex)
                            .map(pair -> pair.second()[tempColumnIndex])
                            .collect(Collectors.toSet());

                    for (Double distinctValue : distinctValues) {
                        long numerator = trainSet.stream()
                                .filter(pair -> pair.first() == tempClassIndex)
                                .filter(pair -> pair.second()[tempColumnIndex] == distinctValue)
                                .count();

                        double valueProbability;
                        if (numerator == 0 || applySmoothingAll) {
                            valueProbability = (double) (numerator + 1) / (classQuantity + distinctValues.size());
                        }
                        else {
                            valueProbability = (double) numerator / (classQuantity);
                        }

                        String valueName = "Column=" + tempColumnIndex + ",Value=" + distinctValue + "|" + className;
                        cache.put(valueName, valueProbability);
                    }

                }
            }
        }
    }

    @Override
    public int predict(double[] input) {
        return 0;
    }
}
