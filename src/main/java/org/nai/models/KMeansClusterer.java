package org.nai.models;

import org.nai.evaluation.EvaluationMetrics;
import org.nai.structures.Centroid;
import org.nai.structures.Pair;
import org.nai.utils.DistanceUtils;

import java.util.*;

public class KMeansClusterer implements Clusterer {
    private final Random random = new Random();

    private int k;
    private int dimensions;

    @Override
    public List<Pair<double[], List<double[]>>> groupClusters(int k, List<double[]> vectors) {
        this.k = k;
        this.dimensions = vectors.getFirst().length;

        Map<Centroid, List<double[]>> clusters = initClusters(vectors);

        double prevRSS = Double.MAX_VALUE;
        double currRSS = EvaluationMetrics.calculateRSS(clusters);

        while (Math.abs(prevRSS - currRSS) > 1e-6) {
            clusters = assignObservations(clusters.keySet(), vectors);
            clusters = recalculateCentroids(clusters);
            prevRSS = currRSS;
            currRSS = EvaluationMetrics.calculateRSS(clusters);
        }

        List<Pair<double[], List<double[]>>> result = new ArrayList<>();
        for (Map.Entry<Centroid, List<double[]>> entry : clusters.entrySet()) {
            result.add(new Pair<>(entry.getKey().getCoordinates(), entry.getValue()));
        }

        return result;
    }

    private Map<Centroid, List<double[]>> initClusters(List<double[]> vectors) {
        Collections.shuffle(vectors);

        Map<Centroid, List<double[]>> clusters = new HashMap<>();
        for (int i = 0; i < k; i++) {
            Centroid c = new Centroid(vectors.get(i).clone());
            clusters.put(c, new ArrayList<>());
        }

        for (int i = 0; i < vectors.size(); i++) {
            Centroid key = new ArrayList<>(clusters.keySet()).get(i % k);
            clusters.get(key).add(vectors.get(i));
        }

        return recalculateCentroids(clusters);
    }

    private Map<Centroid, List<double[]>> assignObservations(Set<Centroid> centroids, List<double[]> vectors) {
        Map<Centroid, List<double[]>> newClusters = new HashMap<>();
        for (Centroid c : centroids) newClusters.put(c, new ArrayList<>());

        for (double[] vec : vectors) {
            Centroid closest = findClosestCentroid(vec, centroids);
            newClusters.get(closest).add(vec);
        }

        return newClusters;
    }

    private Map<Centroid, List<double[]>> recalculateCentroids(Map<Centroid, List<double[]>> oldClusters) {
        Map<Centroid, List<double[]>> newClusters = new HashMap<>();

        for (Map.Entry<Centroid, List<double[]>> entry : oldClusters.entrySet()) {
            List<double[]> vectors = entry.getValue();
            if (vectors.isEmpty()) continue;

            double[] newCenter = new double[dimensions];
            for (double[] vec : vectors) {
                for (int i = 0; i < dimensions; i++) {
                    newCenter[i] += vec[i];
                }
            }
            for (int i = 0; i < dimensions; i++) {
                newCenter[i] /= vectors.size();
            }

            newClusters.put(new Centroid(newCenter), vectors);
        }

        return newClusters;
    }

    private Centroid findClosestCentroid(double[] vector, Set<Centroid> centroids) {
        Centroid best = null;
        double bestDist = Double.MAX_VALUE;

        for (Centroid c : centroids) {
            double dist = DistanceUtils.calculateEuclideanDistance(vector, c.getCoordinates());

            if (dist < bestDist || (dist == bestDist && random.nextBoolean())) {
                bestDist = dist; best = c;
            }
        }

        return best;
    }
}