package org.nai.models;

import java.util.List;
import org.nai.structures.Pair;

public interface Clusterer extends Model {
    List<Pair<double[], List<double[]>>> groupClusters(int k, List<double[]> vectors);
}
