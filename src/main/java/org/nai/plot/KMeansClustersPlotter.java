package org.nai.plot;

import org.nai.structures.Pair;
import org.nai.structures.Vector;
import org.nai.structures.Cluster;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.markers.SeriesMarkers;
import org.knowm.xchart.style.lines.SeriesLines;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class KMeansClustersPlotter {

    public static void plotClusters(List<Cluster> clusters) {
        if (clusters == null || clusters.isEmpty()) {
            System.err.println("No clusters to plot.");
            return;
        }

        int dims = clusters.get(0).getCentroid().getCoordinates().size();
        List<XYChart> charts = new ArrayList<>();

        for (int i = 0; i < dims; i++) {
            for (int j = i + 1; j < dims; j++) {
                XYChart chart = new XYChartBuilder()
                        .width(400).height(400)
                        .title(String.format("Dim %d vs %d", i, j))
                        .xAxisTitle("Feature " + i)
                        .yAxisTitle("Feature " + j)
                        .build();

                chart.getStyler().setMarkerSize(8);

                for (int c = 0; c < clusters.size(); c++) {
                    Cluster cluster = clusters.get(c);

                    List<Vector> members = cluster.getMembers();
                    int tempI = i;
                    double[] x = members.stream().mapToDouble(v -> v.get(tempI)).toArray();
                    int tempJ = j;
                    double[] y = members.stream().mapToDouble(v -> v.get(tempJ)).toArray();

                    chart.addSeries("Cluster " + c, x, y)
                            .setMarker(SeriesMarkers.CIRCLE)
                            .setLineStyle(SeriesLines.NONE);

                    // Plot centroid in black
                    Vector cent = cluster.getCentroid().getCoordinates();
                    double[] cx = { cent.get(i) };
                    double[] cy = { cent.get(j) };
                    chart.addSeries("Centroid " + c, cx, cy)
                            .setMarker(SeriesMarkers.DIAMOND)
                            .setMarkerColor(Color.BLACK)
                            .setLineStyle(SeriesLines.NONE);
                    chart.getStyler().setMarkerSize(12);
                }

                charts.add(chart);
            }
        }

        new SwingWrapper<>(charts).displayChartMatrix();
    }

    public static void plotWCSS(List<Pair<Integer, Double>> kToWcss) {
        if (kToWcss == null || kToWcss.isEmpty()) {
            System.err.println("No WCSS data to plot.");
            return;
        }

        double[] ks   = kToWcss.stream().mapToDouble(Pair::first ).toArray();
        double[] vals = kToWcss.stream().mapToDouble(Pair::second).toArray();

        XYChart chart = new XYChartBuilder()
                .width(600).height(400)
                .title("WCSS vs k")
                .xAxisTitle("k")
                .yAxisTitle("WCSS")
                .build();

        chart.addSeries("WCSS", ks, vals)
                .setMarker(SeriesMarkers.NONE)
                .setLineStyle(SeriesLines.SOLID);

        new SwingWrapper<>(chart).displayChart();
    }
}