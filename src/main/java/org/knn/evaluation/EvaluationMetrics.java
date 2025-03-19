package org.knn.evaluation;

import org.knn.knn.KNearestNeighbours;
import org.knn.data.PrepareDataset;

public class EvaluationMetrics {
    private final int k;
    private final PrepareDataset prepareDataset;

    public EvaluationMetrics(int k, PrepareDataset prepareDataset) {
        this.k = k;
        this.prepareDataset = prepareDataset;
    }

    public void measureAccuracy() {
        int correctPredictionsCount = 0;

        KNearestNeighbours knn = new KNearestNeighbours(k, prepareDataset.getTrainSet());
        for (int i = 0; i < prepareDataset.getTestSet().size(); i++) {
            String answer = knn.run(prepareDataset.getTestSet().get(i));
            if (answer.equals(prepareDataset.getTestLabelsSet().get(i))) {
                correctPredictionsCount++;
            }
        }

        double number = (double) correctPredictionsCount / prepareDataset.getTestSet().size();
        System.out.printf("For the nearest " + k + " observations, the correct predictions count is: %.2f%%%n", number * 100);
    }
}
