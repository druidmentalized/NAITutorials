package org.nai.models;

import java.util.List;

import org.nai.structures.Cluster;
import org.nai.structures.Vector;

public interface Clusterer extends Model {
    List<Cluster> groupClusters(int k, List<Vector> vectors);
}
