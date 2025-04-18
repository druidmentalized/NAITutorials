package org.nai.data;

import org.nai.exceptions.NoFoldersFoundException;
import org.nai.structures.Pair;
import org.nai.utils.FeatureEncoder;
import org.nai.utils.LabelEncoder;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class PrepareDataset {
    public List<Pair<Integer, double[]>> parseDataset(String path, LabelEncoder encoder, FeatureEncoder featureEncoder, boolean isTextCSV) {
        if (path.endsWith(".csv")) {
            if (isTextCSV) {
                return parseTextCSV(path, encoder);
            } else {
                return parseCSV(path, encoder, featureEncoder);
            }
        } else {
            File file = new File(path);
            if (file.isDirectory()) {
                return parseTextDataset(path, encoder);
            } else {
                throw new IllegalArgumentException("Unsupported file format: " + path);
            }
        }
    }

    private List<Pair<Integer, double[]>> parseCSV(String filePath, LabelEncoder labelEncoder, FeatureEncoder featureEncoder) {
        List<Pair<Integer, double[]>> dataset = new ArrayList<>();

        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(filePath))) {
            String line;

            while ((line = bufferedReader.readLine()) != null) {
                String[] tokens = line.split(",");
                int lastIndex = tokens.length - 1;

                // Encode label (last column)
                String labelStr = tokens[lastIndex];
                int encodedLabel = labelEncoder.encode(labelStr);

                // Encode features (all but last)
                double[] vector = new double[lastIndex];
                for (int i = 0; i < lastIndex; i++) {
                    vector[i] = featureEncoder.encode(i, tokens[i]);
                }

                dataset.add(new Pair<>(encodedLabel, vector));
            }

        } catch (IOException e) {
            System.err.println("Reading data went wrong: " + e.getMessage());
        }

        return dataset;
    }

    private List<Pair<Integer, double[]>> parseTextCSV(String filePath, LabelEncoder encoder) {
        List<Pair<Integer, double[]>> dataset = new ArrayList<>();

        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(filePath))) {
            String line;

            while ((line = bufferedReader.readLine()) != null) {
                // Regex splits only on first comma (outside quotes)
                String[] parts = line.split(",", 2);

                if (parts.length != 2) {
                    System.err.println("Skipping invalid line: " + line);
                    continue;
                }

                String labelStr = parts[0].trim();
                String textRaw = parts[1].trim();

                // Remove starting/ending quotes if present
                if (textRaw.startsWith("\"") && textRaw.endsWith("\"")) {
                    textRaw = textRaw.substring(1, textRaw.length() - 1);
                }

                int encodedLabel = encoder.encode(labelStr);
                double[] vector = textToVector(textRaw);

                dataset.add(new Pair<>(encodedLabel, vector));
            }

        } catch (IOException e) {
            System.err.println("Reading data went wrong: " + e.getMessage());
        }

        return dataset;
    }

    private List<Pair<Integer,double[]>> parseTextDataset(String path, LabelEncoder encoder) {
        List<Pair<Integer, double[]>> dataset = new ArrayList<>();

        File mainDir = new File(path);
        File[] languageDirectories = mainDir.listFiles(File::isDirectory);
        if (languageDirectories == null) throw new NoFoldersFoundException("No language folders found in " + mainDir.getAbsolutePath());

        for (File languageDirectory : languageDirectories) {
            String languageName = languageDirectory.getName();
            int encodedLabel = encoder.encode(languageName);

            File[] texts = languageDirectory.listFiles((_, name) -> name.endsWith(".txt"));
            if (texts == null) continue;

            for (File text : texts) {
                try (BufferedReader br = new BufferedReader(new FileReader(text))) {
                    StringBuilder content = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) {
                        if (line.isEmpty()) continue;
                        content.append(line).append(" ");
                    }

                    double[] vector = textToVector(content.toString());
                    dataset.add(new Pair<>(encodedLabel, vector));

                } catch (IOException e) {
                    System.err.println("Error reading file " + text.getName() + ": " + e.getMessage());
                }
            }
        }
        return dataset;
    }

    public SplitDataset trainTestSplit(List<Pair<Integer, double[]>> dataset, double trainRatio) {
        List<Pair<Integer, double[]>> trainSet = new ArrayList<>();
        List<Pair<Integer, double[]>> testSet = new ArrayList<>();

        Map<Integer, List<Pair<Integer, double[]>>> classToSamples = new HashMap<>();
        for (Pair<Integer, double[]> pair : dataset) {
            classToSamples.computeIfAbsent(pair.first(), _ -> new ArrayList<>()).add(pair);
        }

        for (List<Pair<Integer, double[]>> samples : classToSamples.values()) {
            Collections.shuffle(samples);
            int totalClassSize = samples.size();
            int trainCount = Math.max(1, (int) Math.round(totalClassSize * trainRatio));

            trainSet.addAll(samples.subList(0, trainCount));
            testSet.addAll(samples.subList(trainCount, totalClassSize));
        }

        return new SplitDataset(trainSet, testSet);
    }

    public static double[] textToVector(String text) {
        double[] vector = new double[26];
        text = text.toLowerCase().replaceAll("[^a-z]", "");

        for (char ch : text.toCharArray()) {
            if (ch >= 'a' && ch <= 'z') {
                vector[ch - 'a']++;
            }
        }

        int total = text.length();
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= total;
        }

        return vector;
    }
}
