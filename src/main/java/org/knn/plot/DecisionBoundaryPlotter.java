package org.knn.plot;

import org.knn.models.Perceptron;
import org.knn.structures.Pair;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class DecisionBoundaryPlotter {
    private final Perceptron perceptron;
    private final double minX;
    private final double maxX;
    private final int steps;

    private final List<double[]> boundaryPoints;

    public DecisionBoundaryPlotter(Perceptron perceptron, double minX, double maxX, int steps) {
        this.perceptron = perceptron;
        this.minX = minX;
        this.maxX = maxX;
        this.steps = steps;
        this.boundaryPoints = generateBoundaryPoints();
    }

    private List<double[]> generateBoundaryPoints() {
        double[] weights = perceptron.getWeights();
        double threshold = perceptron.getThreshold();

        if (weights.length < 2) {
            throw new IllegalArgumentException("Number of weights must be at least 2 for 2D plot");
        }

        List<double[]> points = new ArrayList<>();
        double stepSize = (maxX - minX) / steps;

        for (int i = 0; i < steps; i++) {
            double x0 = minX + i * stepSize;

            double x1 = (threshold - weights[0] * x0) / weights[1];
            points.add(new double[]{x0, x1});
        }
        return points;
    }

    public void exportBoundaryToCsv(String filePath) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write("x,y\n");
            for (double[] point : boundaryPoints) {
                writer.write(point[0] + "," + point[1] + "\n");
            }
            System.out.println("Boundary points exported to " + filePath);
        } catch (IOException e) {
            System.err.println("Error writing boundary CSV: " + e.getMessage());
        }
    }

    public void asciiPlot(List<Pair<Integer, double[]>> dataset) {
        double minY = Double.MAX_VALUE;
        double maxY = -Double.MAX_VALUE;

        for (double[] bp : boundaryPoints) {
            if (bp[1] < minY) minY = bp[1];
            if (bp[1] > maxY) maxY = bp[1];
        }
        for (Pair<Integer, double[]> pair : dataset) {
            double[] vec = pair.second();
            if (vec.length < 2) continue;
            double y = vec[1];
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        int width = 60;
        int height = 20;

        char[][] grid = new char[height][width];
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                grid[r][c] = ' ';
            }
        }

        Function<Double, Integer> mapX = x -> {
            double ratio = (x - minX) / (maxX - minX);
            return (int) Math.round(ratio * (width - 1));
        };

        double finalMinY = minY;
        double finalMaxY = maxY;
        Function<Double, Integer> mapY = y -> {
            double ratio = (y - finalMinY) / (finalMaxY - finalMinY);
            int row = (int) Math.round((1.0 - ratio) * (height - 1));
            return Math.max(0, Math.min(row, height - 1));
        };

        for (double[] bp : boundaryPoints) {
            double x = bp[0];
            double y = bp[1];
            int col = mapX.apply(x);
            int row = mapY.apply(y);
            if (col >= 0 && col < width && row >= 0 && row < height) {
                grid[row][col] = '*';
            }
        }

        for (Pair<Integer, double[]> pair : dataset) {
            double[] vec = pair.second();
            if (vec.length < 2) continue;
            int col = mapX.apply(vec[0]);
            int row = mapY.apply(vec[1]);
            if (col >= 0 && col < width && row >= 0 && row < height) {
                grid[row][col] = 'o';
            }
        }

        for (int r = 0; r < height; r++) {
            System.out.println(new String(grid[r]));
        }
    }
}
