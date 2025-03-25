package org.knn.evaluation;

import org.knn.data.SplitDataset;
import org.knn.models.Classifier;
import org.knn.models.KNearestNeighbours;
import org.knn.data.PrepareDataset;
import org.knn.structures.Pair;

import java.util.ArrayList;
import java.util.List;

public class EvaluationMetrics {
    Classifier classifier;
    SplitDataset dataset;

    public EvaluationMetrics(Classifier classifier, SplitDataset dataset) {
        this.classifier = classifier;
        this.dataset = dataset;
    }

    public void measureAccuracy() {
        classifier.train(dataset.getTrainSet());

        int correctPredictionsCount = 0;

        List<double[]> testSetVectors = dataset.getTestSetVectors();
        List<Integer> testSetLabels = dataset.getTestSetLabels();

        for (int i = 0; i < testSetVectors.size(); i++) {
            int answer = classifier.predict(testSetVectors.get(i));
            if (answer == testSetLabels.get(i)) correctPredictionsCount++;
        }

        double number = (double) correctPredictionsCount / testSetVectors.size();
        System.out.printf("Evaluated predictions count is: %.2f%%%n", number * 100);
    }
}
