package org.nai.utils;

import java.util.*;

public class FeatureEncoder {
    private final Map<Integer, LabelEncoder> columnEncoders = new HashMap<>();

    public int encode(int columnIndex, String value) {
        columnEncoders.putIfAbsent(columnIndex, new LabelEncoder());
        return columnEncoders.get(columnIndex).encode(value);
    }

    public String decode(int columnIndex, int value) {
        if (!columnEncoders.containsKey(columnIndex)) return null;
        return columnEncoders.get(columnIndex).decode(value);
    }

    public int getDistinctValuesInColumn(int columnIndex) {
        if (!columnEncoders.containsKey(columnIndex)) return 0;
        return columnEncoders.get(columnIndex).getClassesAmount();
    }

    public double[] encodeRow(List<String> row) {
        double[] encoded = new double[row.size()];
        for (int i = 0; i < row.size(); i++) {
            encoded[i] = encode(i, row.get(i));
        }
        return encoded;
    }

    public List<String> decodeRow(double[] encodedRow) {
        List<String> decoded = new ArrayList<>();
        for (int i = 0; i < encodedRow.length; i++) {
            decoded.add(decode(i, (int) encodedRow[i]));
        }
        return decoded;
    }
}
