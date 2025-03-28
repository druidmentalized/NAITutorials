package org.knn.data;

import org.knn.exceptions.NoFoldersFoundException;
import org.knn.structures.Pair;
import org.knn.utils.LabelEncoder;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class PrepareDataset {
    public List<Pair<Integer, double[]>> parseDataset(String path, LabelEncoder encoder) {
        if (path.endsWith(".csv")) {
            return parseCSV(path, encoder);
        } else {
            File file = new File(path);
            if (file.isDirectory()) {
                return parseTextDataset(path, encoder);
            } else {
                throw new IllegalArgumentException("Unsupported file format: " + path);
            }
        }
    }

    private List<Pair<Integer,double[]>> parseCSV(String filePath, LabelEncoder encoder) {
        List<Pair<Integer, double[]>> dataset = new ArrayList<>();

        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(filePath))) {
            String line;

            while ((line = bufferedReader.readLine()) != null) {
                String[] tokens = line.split(",");
                List<Double> vectorList = new ArrayList<>();
                String labelStr = "";

                for (String token : tokens) {
                    try {
                        vectorList.add(Double.parseDouble(token));
                    } catch (NumberFormatException e) {
                        labelStr = token;
                        break;
                    }
                }

                int encodedLabel = encoder.encode(labelStr);

                double[] vector = new double[vectorList.size()];
                for (int i = 0; i < vectorList.size(); i++) {
                    vector[i] = vectorList.get(i);
                }

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
        }

        for (List<Pair<Integer, double[]>> samples : classToSamples.values()) {
            Collections.shuffle(samples);
            int totalClassSize = samples.size();
            int trainCount = Math.max(1, (int) Math.round(totalClassSize * trainRatio));

            trainSet.addAll(samples.subList(0, trainCount));
            testSet.addAll(samples.subList(trainCount, totalClassSize));
        }

        Collections.shuffle(trainSet);
        Collections.shuffle(testSet);

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
