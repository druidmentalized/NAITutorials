package org.nai.utils;

import java.util.HashMap;
import java.util.Map;

public class LabelEncoder {
    private int topIndex = 0;
    private final Map<String, Integer> labelToIndex = new HashMap<>();
    private final Map<Integer, String> indexToLabel = new HashMap<>();

    public int encode(String label) {
        return labelToIndex.computeIfAbsent(label, l -> {
            indexToLabel.put(topIndex, l);
            return topIndex++;
        });
    }

    public String decode(int index) {
        return indexToLabel.get(index);
    }

    public int decode(String label) {
        return labelToIndex.get(label);
    }

    public int getClassesAmount() {
        return labelToIndex.size();
    }
}
