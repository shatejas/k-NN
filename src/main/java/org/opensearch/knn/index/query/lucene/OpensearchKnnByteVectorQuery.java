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
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;

import java.util.Objects;

@Getter
public class OpensearchKnnByteVectorQuery extends KnnByteVectorQuery {

    private final int kCandidates;

    public OpensearchKnnByteVectorQuery(final String field, final byte[] target, int k, Integer efSearch) {
        this(field, target, k, null, efSearch);
    }

    public OpensearchKnnByteVectorQuery(String field, byte[] target, int k, Query filter, Integer efSearch) {
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
        OpensearchKnnByteVectorQuery that = (OpensearchKnnByteVectorQuery) o;
        return kCandidates == that.kCandidates;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        return result + Objects.hash(kCandidates);
    }
}
