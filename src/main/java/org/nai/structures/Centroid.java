package org.nai.structures;

import java.util.Arrays;

public class Centroid {
    private final Vector coordinates;

    public Centroid(Vector coordinates) {
        this.coordinates = coordinates;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Centroid that)) return false;
        return Arrays.equals(coordinates.getData(), that.coordinates.getData());
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(coordinates.getData());
    }

    public Vector getCoordinates() {
        return coordinates;
    }
}
