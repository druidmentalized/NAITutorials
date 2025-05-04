package org.nai.models;

import org.nai.data.Dataset;
import org.nai.structures.Vector;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class NaiveBayes implements Classifier {

    private final int classesAmount;
    private final boolean applySmoothingAll;
    Map<String, Double> cache = new HashMap<>();

    public NaiveBayes(int classesAmount, boolean applySmoothingAll) {
        this.classesAmount = classesAmount;
        this.applySmoothingAll = applySmoothingAll;
    }

    @Override
    public void train(Dataset trainSet) {
        cache.clear();

        int entriesAmount = trainSet.size();
        for (int classIndex = 0; classIndex < classesAmount; classIndex++) {
            String className = "Class=" + classIndex;
            int tempClassIndex = classIndex;
            long classQuantity = trainSet.getData().stream().filter(pair -> pair.first() == tempClassIndex).count();

            findPriori(classQuantity, entriesAmount, className);
            findPosteriori(trainSet, tempClassIndex, classQuantity, className);
        }
    }

    private void findPriori(long classQuantity, int entriesAmount, String className) {
        double classPriorProbability = (double) classQuantity / entriesAmount;
        cache.put(className, classPriorProbability);
    }

    private void findPosteriori(Dataset trainSet, int tempClassIndex, long classQuantity, String className) {
        List<org.nai.structures.Pair<Integer, Vector>> classSamples = trainSet.getData().stream()
                .filter(pair -> pair.first() == tempClassIndex)
                .toList();

        int columnsAmount = classSamples.getFirst().second().size();

        for (int columnIndex = 0; columnIndex < columnsAmount; columnIndex++) {
            int tempColumnIndex = columnIndex;
            Set<Double> distinctValues = classSamples.stream()
                    .map(pair -> pair.second().get(tempColumnIndex))
                    .collect(Collectors.toSet());

            for (Double distinctValue : distinctValues) {
                long numerator = classSamples.stream()
                        .filter(pair -> pair.second().get(tempColumnIndex) == distinctValue)
                        .count();

                double valueProbability;
                if (numerator == 0 || applySmoothingAll) {
                    valueProbability = (double) (numerator + 1) / (classQuantity + distinctValues.size());
                }
                else {
                    valueProbability = (double) numerator / (classQuantity);
                }

                String valueName = "Column=" + columnIndex + ",Value=" + distinctValue + "|" + className;
                cache.put(valueName, valueProbability);
            }

        }
    }

    @Override
    public int predict(Vector input) {
        double maxLogProbability = Double.NEGATIVE_INFINITY;
        int bestClass = -1;

        for (int classIndex = 0; classIndex < classesAmount; classIndex++) {
            String className = "Class=" + classIndex;
            double logProb = Math.log(cache.get(className));

            for (int col = 0; col < input.size(); col++) {
                String key = "Column=" + col + ",Value=" + input.get(col) + "|" + className;
                double condProb = cache.getOrDefault(key, 1e-9);
                logProb += Math.log(condProb);
            }

            if (logProb > maxLogProbability) {
                maxLogProbability = logProb;
                bestClass = classIndex;
            }
        }

        return bestClass;
    }
}
