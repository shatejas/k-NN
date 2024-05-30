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
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;

import java.util.Objects;

@Getter
public class OpensearchDiversifyingChildrenFloatKnnVectorQuery extends DiversifyingChildrenFloatKnnVectorQuery {

    private final int kCandidates;

    public OpensearchDiversifyingChildrenFloatKnnVectorQuery(
        String field,
        float[] query,
        Query childFilter,
        int k,
        BitSetProducer parentsFilter,
        final Integer efSearch
    ) {
        super(field, query, childFilter, efSearch == null ? k : efSearch, parentsFilter);
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
        OpensearchDiversifyingChildrenFloatKnnVectorQuery that = (OpensearchDiversifyingChildrenFloatKnnVectorQuery) o;
        return kCandidates == that.kCandidates;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        return result + Objects.hash(kCandidates);
    }
}
