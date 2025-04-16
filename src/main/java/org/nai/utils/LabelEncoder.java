package org.nai.utils;

import java.util.HashMap;
import java.util.Map;

public class LabelEncoder {
    private int topIndex = 0;
    private final Map<String, Integer> labelToIndex = new HashMap<>();
    private final Map<Integer, String> indexToLabel = new HashMap<>();

    public int encode(String label) {
        if (!labelToIndex.containsKey(label)) {
            labelToIndex.put(label, topIndex);
            indexToLabel.put(topIndex++, label);
        }
        return labelToIndex.get(label);
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
