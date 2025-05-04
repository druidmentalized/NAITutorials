package org.nai.data;

import org.nai.exceptions.NoFoldersFoundException;
import org.nai.exceptions.UnsupportedFileFormatException;
import org.nai.structures.Pair;
import org.nai.structures.Vector;
import org.nai.utils.FeatureEncoder;
import org.nai.utils.LabelEncoder;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Collections;

public class PrepareDataset {
    public Dataset parseDataset(
            String path,
            LabelEncoder encoder,
            FeatureEncoder featureEncoder,
            boolean isTextCSV,
            boolean encodeFeatures
    ) {
        if (path.endsWith(".csv")) {
            if (isTextCSV) {
                return parseTextCSV(path, encoder);
            } else {
                return parseCSV(path, encoder, featureEncoder, encodeFeatures);
            }
        } else {
            File file = new File(path);
            if (file.isDirectory()) {
                return parseTextDataset(path, encoder);
            } else {
                throw new UnsupportedFileFormatException("Unsupported file format: " + path);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //                    Numeric OR Categorical CSV parser
    // ─────────────────────────────────────────────────────────────────────
    private Dataset parseCSV(
            String filePath,
            LabelEncoder labelEncoder,
            FeatureEncoder featureEncoder,
            boolean encodeFeatures
    ) {
        Dataset dataset = new Dataset();

        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split(",");
                int lastIndex = tokens.length - 1;

                int encodedLabel = labelEncoder.encode(tokens[lastIndex].trim());

                double[] vectorArr = new double[lastIndex];
                for (int i = 0; i < lastIndex; i++) {
                    String tok = tokens[i].trim();
                    if (encodeFeatures) {
                        vectorArr[i] = featureEncoder.encode(i, tok);
                    } else {
                        vectorArr[i] = Double.parseDouble(tok);
                    }
                }

                dataset.add(new Pair<>(encodedLabel, new Vector(vectorArr)));
            }
        } catch (IOException e) {
            System.err.println("Reading data went wrong: " + e.getMessage());
        }

        return dataset;
    }

    // ─────────────────────────────────────────────────────────────────────
    //                     Text‐only CSV parser (Label, "Long text…")
    // ─────────────────────────────────────────────────────────────────────
    private Dataset parseTextCSV(
            String filePath,
            LabelEncoder encoder
    ) {
        Dataset dataset = new Dataset();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int lineNumber = 1;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",", 2);
                if (parts.length != 2) {
                    System.err.printf("Line %d in %s is invalid CSV format: '%s'%n", lineNumber, filePath, line);
                    continue;
                }
                String labelStr = parts[0].trim();
                String textRaw = parts[1].trim();
                if (textRaw.startsWith("\"") && textRaw.endsWith("\"")) {
                    textRaw = textRaw.substring(1, textRaw.length() - 1);
                }
                int lbl = encoder.encode(labelStr);
                Vector vec = textToVector(textRaw);
                dataset.add(new Pair<>(lbl, vec));
                lineNumber++;
            }
        } catch (IOException e) {
            System.err.println("Reading data went wrong: " + e.getMessage());
        }

        return dataset;
    }

    // ─────────────────────────────────────────────────────────────────────
    //                Directory‐of‐.txt‐files parser by subfolder
    // ─────────────────────────────────────────────────────────────────────
    private Dataset parseTextDataset(
            String path,
            LabelEncoder encoder
    ) {
        Dataset dataset = new Dataset();
        File mainDir = new File(path);
        File[] folders = mainDir.listFiles(File::isDirectory);
        if (folders == null || folders.length == 0) throw new NoFoldersFoundException(path);

        for (File langDir : folders) {
            processTextFolder(langDir, encoder, dataset);
        }

        return dataset;
    }

    private void processTextFolder(File folder, LabelEncoder encoder, Dataset dataset) {
        int lbl = encoder.encode(folder.getName());
        File[] files = folder.listFiles((_, n) -> n.endsWith(".txt"));
        if (files == null) return;
        for (File file : files) {
            String content = readFileContent(file);
            if (!content.isEmpty()) {
                dataset.add(new Pair<>(lbl, textToVector(content)));
            }
        }
    }

    private String readFileContent(File file) {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.isEmpty()) sb.append(line).append(' ');
            }
        } catch (IOException ex) {
            System.err.println("Error reading " + file.getName() + ": " + ex.getMessage());
        }
        return sb.toString();
    }

    // ─────────────────────────────────────────────────────────────────────
    //                      Stratified train/test split
    // ─────────────────────────────────────────────────────────────────────
    public SplitDataset trainTestSplit(Dataset wholeDataset, double trainRatio) {
        Dataset trainSet = new Dataset();
        Dataset testSet = new Dataset();

        Map<Integer, Dataset> byClass = new HashMap<>();
        for (Pair<Integer, Vector> p : wholeDataset.getData()) {
            byClass.computeIfAbsent(p.first(), _ -> new Dataset()).add(p);
        }

        for (Dataset bucket : byClass.values()) {
            List<Pair<Integer, Vector>> data = bucket.getData();
            Collections.shuffle(data);
            int cut = Math.max(1, (int) Math.round(data.size() * trainRatio));

            trainSet.addAll(data.subList(0, cut));
            testSet.addAll(data.subList(cut, data.size()));
        }
        return new SplitDataset(trainSet, testSet);
    }

    // ─────────────────────────────────────────────────────────────────────
    //                         Text → 26‐dim letter‐freq vector
    // ─────────────────────────────────────────────────────────────────────
    public static Vector textToVector(String text) {
        double[] data = new double[26];
        text = text.toLowerCase().replaceAll("[^a-z]", "");

        for (char c : text.toCharArray()) {
            if (c >= 'a' && c <= 'z') {
                data[c - 'a']++;
            }
        }

        double len = text.length();
        if (len > 0) {
            for (int i = 0; i < 26; i++) {
                data[i] /= len;
            }
        }

        return new Vector(data);
    }
}