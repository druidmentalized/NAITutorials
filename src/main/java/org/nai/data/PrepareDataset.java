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
    public List<Pair<Integer, double[]>> parseDataset(
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
                throw new IllegalArgumentException("Unsupported file format: " + path);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //                    Numeric OR Categorical CSV parser
    // ─────────────────────────────────────────────────────────────────────
    private List<Pair<Integer, double[]>> parseCSV(
            String filePath,
            LabelEncoder labelEncoder,
            FeatureEncoder featureEncoder,
            boolean encodeFeatures
    ) {
        List<Pair<Integer, double[]>> dataset = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split(",");
                int lastIndex = tokens.length - 1;

                int encodedLabel = labelEncoder.encode(tokens[lastIndex].trim());

                double[] vector = new double[lastIndex];
                for (int i = 0; i < lastIndex; i++) {
                    String tok = tokens[i].trim();
                    if (encodeFeatures) {
                        vector[i] = featureEncoder.encode(i, tok);
                    } else {
                        vector[i] = Double.parseDouble(tok);
                    }
                }

                dataset.add(new Pair<>(encodedLabel, vector));
            }
        } catch (IOException e) {
            System.err.println("Reading data went wrong: " + e.getMessage());
        }

        return dataset;
    }

    // ─────────────────────────────────────────────────────────────────────
    //                     Text‐only CSV parser (Label, "Long text…")
    // ─────────────────────────────────────────────────────────────────────
    private List<Pair<Integer, double[]>> parseTextCSV(
            String filePath,
            LabelEncoder encoder
    ) {
        List<Pair<Integer, double[]>> dataset = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",", 2);
                if (parts.length != 2) {
                    System.err.println("Skipping invalid line: " + line);
                    continue;
                }
                String labelStr = parts[0].trim();
                String textRaw = parts[1].trim();
                if (textRaw.startsWith("\"") && textRaw.endsWith("\"")) {
                    textRaw = textRaw.substring(1, textRaw.length() - 1);
                }
                int lbl = encoder.encode(labelStr);
                double[] vec = textToVector(textRaw);
                dataset.add(new Pair<>(lbl, vec));
            }
        } catch (IOException e) {
            System.err.println("Reading data went wrong: " + e.getMessage());
        }

        return dataset;
    }

    // ─────────────────────────────────────────────────────────────────────
    //                Directory‐of‐.txt‐files parser by subfolder
    // ─────────────────────────────────────────────────────────────────────
    private List<Pair<Integer, double[]>> parseTextDataset(
            String path,
            LabelEncoder encoder
    ) {
        List<Pair<Integer, double[]>> dataset = new ArrayList<>();
        File mainDir = new File(path);
        File[] folders = mainDir.listFiles(File::isDirectory);
        if (folders == null) throw new NoFoldersFoundException(path);

        for (File langDir : folders) {
            int lbl = encoder.encode(langDir.getName());
            File[] files = langDir.listFiles((_, n) -> n.endsWith(".txt"));
            if (files == null) continue;
            for (File f : files) {
                try (BufferedReader br = new BufferedReader(new FileReader(f))) {
                    StringBuilder sb = new StringBuilder();
                    String l;
                    while ((l = br.readLine()) != null) {
                        if (!l.isEmpty()) sb.append(l).append(' ');
                    }
                    dataset.add(new Pair<>(lbl, textToVector(sb.toString())));
                } catch (IOException ex) {
                    System.err.println("Error reading " + f.getName() + ": " + ex.getMessage());
                }
            }
        }

        return dataset;
    }

    // ─────────────────────────────────────────────────────────────────────
    //                      Stratified train/test split
    // ─────────────────────────────────────────────────────────────────────
    public SplitDataset trainTestSplit(List<Pair<Integer, double[]>> data, double trainRatio) {
        List<Pair<Integer,double[]>> train = new ArrayList<>();
        List<Pair<Integer,double[]>> test  = new ArrayList<>();

        Map<Integer,List<Pair<Integer,double[]>>> byClass = new HashMap<>();
        for (Pair<Integer,double[]> p : data) {
            byClass.computeIfAbsent(p.first(), _ -> new ArrayList<>()).add(p);
        }

        for (List<Pair<Integer,double[]>> bucket : byClass.values()) {
            Collections.shuffle(bucket);
            int cut = Math.max(1, (int)Math.round(bucket.size() * trainRatio));
            train.addAll(bucket.subList(0, cut));
            test .addAll(bucket.subList(cut, bucket.size()));
        }
        return new SplitDataset(train, test);
    }

    // ─────────────────────────────────────────────────────────────────────
    //                         Text → 26‐dim letter‐freq vector
    // ─────────────────────────────────────────────────────────────────────
    public static double[] textToVector(String text) {
        double[] vec = new double[26];
        text = text.toLowerCase().replaceAll("[^a-z]", "");
        for (char c : text.toCharArray()) {
            vec[c - 'a']++;
        }
        double len = text.length();
        for (int i = 0; i < 26; i++) {
            vec[i] /= len;
        }
        return vec;
    }
}