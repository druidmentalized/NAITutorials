package org.nai.structures;

import java.util.Arrays;

public class Centroid {
    private final double[] coordinates;

    public Centroid(double[] coordinates) {
        this.coordinates = coordinates;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Centroid that)) return false;
        return Arrays.equals(coordinates, that.coordinates);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(coordinates);
    }

    public double[] getCoordinates() {
        return coordinates;
    }
}
