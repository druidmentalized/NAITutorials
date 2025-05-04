package org.nai.models;

import org.nai.evaluation.EvaluationMetrics;
import org.nai.structures.Cluster;
import org.nai.structures.Vector;

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;

public class KMeansClusterer implements Clusterer {
    private int k;
    @Override
    public List<Cluster> groupClusters(int k, List<Vector> vectors) {
        this.k = k;

        List<Cluster> clusters = initClusters(vectors);

        double prevRSS = Double.MAX_VALUE;
        double currRSS = EvaluationMetrics.computeWCSS(clusters);

        while (Math.abs(prevRSS - currRSS) > 1e-6) {
            assignObservations(clusters, vectors);
            recalculateCentroids(clusters);
            prevRSS = currRSS;
            currRSS = EvaluationMetrics.computeWCSS(clusters);
        }

        return clusters;
    }

    private List<Cluster> initClusters(List<Vector> vectors) {
        Collections.shuffle(vectors);
        List<Cluster> clusters = new ArrayList<>();

        for (int i = 0; i < k; i++) {
            clusters.add(new Cluster(null, new ArrayList<>()));
        }

        for (int i = 0; i < vectors.size(); i++) {
            clusters.get(i % k).addMember(vectors.get(i));
        }

        recalculateCentroids(clusters);

        return clusters;
    }

    private void assignObservations(List<Cluster> clusters, List<Vector> vectors) {
        clusters.forEach(cluster -> cluster.getMembers().clear());

        for (Vector vector : vectors) {
            Cluster bestCluster = findClosestCluster(vector, clusters);
            bestCluster.addMember(vector);
        }
    }

    private void recalculateCentroids(List<Cluster> clusters) {
        clusters.forEach(Cluster::recalculateCentroid);
    }

    private Cluster findClosestCluster(Vector vector, List<Cluster> clusters) {
        Cluster best = null;
        double bestDistance = Double.MAX_VALUE;

        for (Cluster cluster : clusters) {
            double distance = vector.squaredDistanceTo(cluster.getCentroid().getCoordinates());
            if (distance < bestDistance) {
                best = cluster;
                bestDistance = distance;
            }
        }

        return best;
    }
}