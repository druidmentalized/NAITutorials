package org.nai.data;

import org.nai.structures.Pair;
import org.nai.structures.Vector;

import java.util.ArrayList;
import java.util.List;

public class Dataset {
    private final List<Pair<Integer, Vector>> data;

    public Dataset() {
        this.data = new ArrayList<>();
    }

    public Dataset(List<Pair<Integer, Vector>> data) {
        this.data = data;
    }

    public void add(Pair<Integer, Vector> pair) {
        data.add(pair);
    }

    public void addAll(List<Pair<Integer, Vector>> entries) {
        data.addAll(entries);
    }

    public int size() {
        return data.size();
    }

    public List<Pair<Integer, Vector>> getData() {
        return data;
    }

    public List<Integer> getLabels() {
        return data.stream().map((Pair::first)).toList();
    }

    public List<Vector> getVectors() {
        return data.stream().map((Pair::second)).toList();
    }
}
