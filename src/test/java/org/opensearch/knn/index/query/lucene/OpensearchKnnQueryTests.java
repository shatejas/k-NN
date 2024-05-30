/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucene;

import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.opensearch.knn.KNNTestCase;

import java.util.Arrays;

public class OpensearchKnnQueryTests extends KNNTestCase {

    private static final String FIELD_NAME = "field";
    private static final String FILTER_FILED_NAME = "foo";
    private static final String FILTER_FILED_VALUE = "fooval";
    private static final float[] FLOAT_VECTOR_QUERY = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    private static final byte[] BYTE_VECTOR_QUERY = new byte[] { 1, 2, 3, 4, 5 };
    private static final Query FILTER_QUERY = new TermQuery(new Term(FILTER_FILED_NAME, FILTER_FILED_VALUE));
    private static final int K = 10;
    private static final Integer EF = 100;

    public void testFloatVectorQuery() {
        OpensearchKnnFloatVectorQuery q1 = new OpensearchKnnFloatVectorQuery(FIELD_NAME, FLOAT_VECTOR_QUERY, K, EF);
        assertEquals((int) EF, q1.getK());
        assertEquals(K, q1.getKCandidates());

        OpensearchKnnFloatVectorQuery q2 = new OpensearchKnnFloatVectorQuery(FIELD_NAME, FLOAT_VECTOR_QUERY, K, EF);
        assertEquals(q1, q2);
        assertEquals(q1.hashCode(), q2.hashCode());

        OpensearchKnnFloatVectorQuery q3 = new OpensearchKnnFloatVectorQuery(FIELD_NAME, FLOAT_VECTOR_QUERY, K, null);
        assertNotEquals(q3, q2);
        assertNotEquals(q3.hashCode(), q2.hashCode());
        assertEquals(K, q3.getK());
        assertEquals(K, q3.getKCandidates());

        OpensearchKnnFloatVectorQuery q4 = new OpensearchKnnFloatVectorQuery(FIELD_NAME, FLOAT_VECTOR_QUERY, K, FILTER_QUERY, EF);
        assertNotEquals(q4, q2);
        assertNotEquals(q4, q3);

        OpensearchKnnFloatVectorQuery q5 = new OpensearchKnnFloatVectorQuery(FIELD_NAME, FLOAT_VECTOR_QUERY, K, FILTER_QUERY, EF);
        assertEquals(q5, q4);
    }

    public void testDiversifyFloatVectorQuery() {
        OpensearchDiversifyingChildrenFloatKnnVectorQuery q1 = new OpensearchDiversifyingChildrenFloatKnnVectorQuery(
            FIELD_NAME,
            FLOAT_VECTOR_QUERY,
            null,
            K,
            null,
            EF
        );
        assertEquals((int) EF, q1.getK());
        assertEquals(K, q1.getKCandidates());

        OpensearchDiversifyingChildrenFloatKnnVectorQuery q2 = new OpensearchDiversifyingChildrenFloatKnnVectorQuery(
            FIELD_NAME,
            FLOAT_VECTOR_QUERY,
            null,
            K,
            null,
            EF
        );
        assertEquals(q1, q2);
        assertEquals(q1.hashCode(), q2.hashCode());

        OpensearchDiversifyingChildrenFloatKnnVectorQuery q3 = new OpensearchDiversifyingChildrenFloatKnnVectorQuery(
            FIELD_NAME,
            FLOAT_VECTOR_QUERY,
            null,
            K,
            null,
            null
        );
        assertNotEquals(q3, q2);
        assertNotEquals(q3.hashCode(), q2.hashCode());
        assertEquals(K, q3.getK());
        assertEquals(K, q3.getKCandidates());

        OpensearchDiversifyingChildrenFloatKnnVectorQuery q4 = new OpensearchDiversifyingChildrenFloatKnnVectorQuery(
            FIELD_NAME,
            FLOAT_VECTOR_QUERY,
            FILTER_QUERY,
            K,
            null,
            EF
        );
        assertNotEquals(q4, q2);
        assertNotEquals(q4, q3);

        OpensearchDiversifyingChildrenFloatKnnVectorQuery q5 = new OpensearchDiversifyingChildrenFloatKnnVectorQuery(
            FIELD_NAME,
            FLOAT_VECTOR_QUERY,
            FILTER_QUERY,
            K,
            null,
            EF
        );
        assertEquals(q5, q4);
    }

    public void testByteVectorQuery() {
        OpensearchKnnByteVectorQuery q1 = new OpensearchKnnByteVectorQuery(FIELD_NAME, BYTE_VECTOR_QUERY, K, EF);
        assertEquals((int) EF, q1.getK());
        assertEquals(K, q1.getKCandidates());

        OpensearchKnnByteVectorQuery q2 = new OpensearchKnnByteVectorQuery(FIELD_NAME, BYTE_VECTOR_QUERY, K, EF);
        assertEquals(q1, q2);
        assertEquals(q1.hashCode(), q2.hashCode());

        OpensearchKnnByteVectorQuery q3 = new OpensearchKnnByteVectorQuery(FIELD_NAME, BYTE_VECTOR_QUERY, K, null);
        assertNotEquals(q3, q2);
        assertNotEquals(q3.hashCode(), q2.hashCode());
        assertEquals(K, q3.getK());
        assertEquals(K, q3.getKCandidates());

        OpensearchKnnByteVectorQuery q4 = new OpensearchKnnByteVectorQuery(FIELD_NAME, BYTE_VECTOR_QUERY, K, FILTER_QUERY, EF);
        assertNotEquals(q4, q2);
        assertNotEquals(q4, q3);

        OpensearchKnnByteVectorQuery q5 = new OpensearchKnnByteVectorQuery(FIELD_NAME, BYTE_VECTOR_QUERY, K, FILTER_QUERY, EF);
        assertEquals(q5, q4);
    }

    public void testDiversifyByteVectorQuery() {
        OpensearchDiversifyingChildrenByteKnnVectorQuery q1 = new OpensearchDiversifyingChildrenByteKnnVectorQuery(
            FIELD_NAME,
            BYTE_VECTOR_QUERY,
            null,
            K,
            null,
            EF
        );
        assertEquals((int) EF, q1.getK());
        assertEquals(K, q1.getKCandidates());

        OpensearchDiversifyingChildrenByteKnnVectorQuery q2 = new OpensearchDiversifyingChildrenByteKnnVectorQuery(
            FIELD_NAME,
            BYTE_VECTOR_QUERY,
            null,
            K,
            null,
            EF
        );
        assertEquals(q1, q2);
        assertEquals(q1.hashCode(), q2.hashCode());

        OpensearchDiversifyingChildrenByteKnnVectorQuery q3 = new OpensearchDiversifyingChildrenByteKnnVectorQuery(
            FIELD_NAME,
            BYTE_VECTOR_QUERY,
            null,
            K,
            null,
            null
        );
        assertNotEquals(q3, q2);
        assertNotEquals(q3.hashCode(), q2.hashCode());
        assertEquals(K, q3.getK());
        assertEquals(K, q3.getKCandidates());

        OpensearchDiversifyingChildrenByteKnnVectorQuery q4 = new OpensearchDiversifyingChildrenByteKnnVectorQuery(
            FIELD_NAME,
            BYTE_VECTOR_QUERY,
            FILTER_QUERY,
            K,
            null,
            EF
        );
        assertNotEquals(q4, q2);
        assertNotEquals(q4, q3);

        OpensearchDiversifyingChildrenByteKnnVectorQuery q5 = new OpensearchDiversifyingChildrenByteKnnVectorQuery(
            FIELD_NAME,
            BYTE_VECTOR_QUERY,
            FILTER_QUERY,
            K,
            null,
            EF
        );
        assertEquals(q5, q4);
    }

    public void testMergeDocOverride() {

        TopDocs[] topDocs = buildTopDocs(
            new ScoreDoc[] { scoreDoc(2, 2.0f), scoreDoc(1, 1.0f) },
            new ScoreDoc[] { scoreDoc(4, 4.0f), scoreDoc(3, 3.0f) },
            new ScoreDoc[] { scoreDoc(6, 6.0f), scoreDoc(5, 5.0f) }
        );

        int k = 3;
        ScoreDoc[] expectedScoreDocs = new ScoreDoc[] { scoreDoc(6, 6.0f), scoreDoc(5, 5.0f), scoreDoc(4, 4.0f) };
        TotalHits expectedTotalHits = new TotalHits(6, TotalHits.Relation.EQUAL_TO);

        OpensearchKnnFloatVectorQuery q1 = new OpensearchKnnFloatVectorQuery(FIELD_NAME, FLOAT_VECTOR_QUERY, k, EF);
        TopDocs resultTopDoc = q1.mergeLeafResults(topDocs);
        asserScoreDocs(expectedScoreDocs, resultTopDoc.scoreDocs);
        assertEquals(expectedTotalHits, resultTopDoc.totalHits);

        OpensearchDiversifyingChildrenFloatKnnVectorQuery q2 = new OpensearchDiversifyingChildrenFloatKnnVectorQuery(
            FIELD_NAME,
            FLOAT_VECTOR_QUERY,
            null,
            k,
            null,
            EF
        );
        ;
        resultTopDoc = q2.mergeLeafResults(topDocs);
        asserScoreDocs(expectedScoreDocs, resultTopDoc.scoreDocs);
        assertEquals(expectedTotalHits, resultTopDoc.totalHits);

        OpensearchKnnByteVectorQuery q3 = new OpensearchKnnByteVectorQuery(FIELD_NAME, BYTE_VECTOR_QUERY, k, EF);
        resultTopDoc = q3.mergeLeafResults(topDocs);
        asserScoreDocs(expectedScoreDocs, resultTopDoc.scoreDocs);
        assertEquals(expectedTotalHits, resultTopDoc.totalHits);

        OpensearchDiversifyingChildrenByteKnnVectorQuery q4 = new OpensearchDiversifyingChildrenByteKnnVectorQuery(
            FIELD_NAME,
            BYTE_VECTOR_QUERY,
            null,
            k,
            null,
            EF
        );
        resultTopDoc = q4.mergeLeafResults(topDocs);
        asserScoreDocs(expectedScoreDocs, resultTopDoc.scoreDocs);
        assertEquals(expectedTotalHits, resultTopDoc.totalHits);
    }

    private void asserScoreDocs(ScoreDoc[] expectedScoreDocs, ScoreDoc[] scoreDocs) {
        assertEquals(expectedScoreDocs.length, scoreDocs.length);
        for (int i = 0; i < expectedScoreDocs.length; i++) {
            assertEquals(expectedScoreDocs[i].score, scoreDocs[i].score, 0.001);
            assertEquals(expectedScoreDocs[i].doc, scoreDocs[i].doc);
        }
    }

    private ScoreDoc scoreDoc(int id, float score) {
        return new ScoreDoc(id, score);
    }

    private TopDocs[] buildTopDocs(ScoreDoc[]... scoreDocs) {
        return Arrays.stream(scoreDocs)
            .map(scoreDoc -> new TopDocs(new TotalHits(scoreDoc.length, TotalHits.Relation.EQUAL_TO), scoreDoc))
            .toArray(TopDocs[]::new);
    }
}
