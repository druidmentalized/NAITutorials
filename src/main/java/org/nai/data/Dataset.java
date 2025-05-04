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

    public List<Pair<Integer, Vector>> getData() {
        return data;
    }

    public void add(Pair<Integer, Vector> pair) {
        data.add(pair);
    }

    public int size() {
        return data.size();
    }
}
