package org.nai.structures;

import java.util.Arrays;

public class Vector {
    private final double[] data;

    public Vector(double[] data) {
        this.data = data.clone();
    }

    public int size() {
        return data.length;
    }

    public double get(int index) {
        checkBounds(index);
        return data[index];
    }

    public void set(int index, double value) {
        checkBounds(index);
        data[index] = value;
    }

    public double[] getData() {
        return data.clone();
    }

    public Vector add(Vector other) {
        checkDimension(other);
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] + other.data[i];
        }
        return new Vector(result);
    }

    public Vector scale(double scalar) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] * scalar;
        }
        return new Vector(result);
    }

    public double dot(Vector other) {
        checkDimension(other);
        double result = 0;
        for (int i = 0; i < data.length; i++) {
            result += data[i] * other.data[i];
        }
        return result;
    }

    public double norm() {
        return Math.sqrt(dot(this));
    }

    public double distanceTo(Vector other) {
        return Math.sqrt(squaredDistanceTo(other));
    }

    public double squaredDistanceTo(Vector other) {
        checkDimension(other);
        double sum = 0;
        for (int i = 0; i < data.length; i++) {
            double d = data[i] - other.data[i];
            sum += d * d;
        }
        return sum;
    }

    private void checkDimension(Vector other) {
        if (data.length != other.data.length) {
            throw new IllegalArgumentException("Vectors must be the same length");
        }
    }

    private void checkBounds(int index) {
        if (index < 0 || index >= data.length) {
            throw new IndexOutOfBoundsException("Invalid index: " + index);
        }
    }

    @Override
    public String toString() {
        return Arrays.toString(data);
    }
}