/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.query.lucene;

import lombok.Getter;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;

import java.util.Objects;

@Getter
public class OpensearchKnnFloatVectorQuery extends KnnFloatVectorQuery {

    private final int kCandidates;

    public OpensearchKnnFloatVectorQuery(final String field, final float[] target, int k, Integer efSearch) {
        this(field, target, k, null, efSearch);
    }

    public OpensearchKnnFloatVectorQuery(String field, float[] target, int k, Query filter, Integer efSearch) {
        super(field, target, efSearch == null ? k : efSearch, filter);
        this.kCandidates = k;
    }

    @Override
    protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
        return TopDocs.merge(this.kCandidates, perLeafResults);
    }

    @Override
    public String toString(String field) {
        return getClass().getSimpleName() + ":" + this.field + ",...][" + kCandidates + "]";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!super.equals(o)) return false;
        OpensearchKnnFloatVectorQuery that = (OpensearchKnnFloatVectorQuery) o;
        return kCandidates == that.kCandidates;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        return result + Objects.hash(kCandidates);
    }
}
