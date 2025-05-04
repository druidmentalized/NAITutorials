package org.nai.utils;

public class DistanceUtils {
    public static double calculateSquaredEuclideanDistance(double[] vec1, double[] vec2) {
        double result = 0;

        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Vectors must be the same dimension");
        }

        for (int i = 0; i < vec1.length; i++) {
            result += Math.pow((vec1[i] - vec2[i]), 2);
        }

        return result;
    }

    public static double calculateEuclideanDistance(double[] vec1, double[] vec2) {
        return Math.sqrt(calculateSquaredEuclideanDistance(vec1, vec2));
    }
}
