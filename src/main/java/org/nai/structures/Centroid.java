package org.nai.structures;

import java.util.Arrays;

public record Centroid(Vector coordinates) {

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Centroid(Vector coordinates1))) return false;
        return Arrays.equals(coordinates.data(), coordinates1.data());
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(coordinates.data());
    }
}
