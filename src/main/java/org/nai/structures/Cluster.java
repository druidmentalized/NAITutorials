package org.nai.structures;

import java.util.List;

public class Cluster {
    private Centroid centroid;
    private final List<Vector> members;

    public Cluster(Centroid centroid, List<Vector> members) {
        this.centroid = centroid;
        this.members = members;
    }

    public void recalculateCentroid() {
        int dimensions = members.getFirst().size();
        double[] vectorArr = new double[dimensions];

        for (int i = 0; i < dimensions; i++) {
            for (Vector member : members) {
                vectorArr[i] += member.get(i);
            }
            vectorArr[i] /= members.size();
        }

        this.centroid = new Centroid(new Vector(vectorArr));
    }

    public void addMember(Vector member) {
        members.add(member);
    }

    public Centroid getCentroid() {
        return centroid;
    }

    public List<Vector> getMembers() {
        return members;
    }
}
