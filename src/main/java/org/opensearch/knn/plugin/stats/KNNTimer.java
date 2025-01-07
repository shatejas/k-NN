/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.opensearch.search.profile.Timer;

@Getter
@RequiredArgsConstructor
public enum KNNTimer {

    FILTER_WEIGHT_TIME("filter_weight", new Timer()),
    FILTER_SCORER_TIME("filter_scorer", new Timer()),
    EXACT_SEARCH_TIME("exact_search", new Timer()),
    FILTER_ID_SELECTOR_TIME("filter_id_selector", new Timer()),
    ANN_TIME("ann_search", new Timer());

    private final String name;
    private final Timer timer;

    public void start() {
        timer.start();
    }

    public void stop() {
        timer.stop();
    }

    public long average() {
        if (timer.getCount() > 1) {
            return timer.getApproximateTiming() / timer.getCount();
        }
        return timer.getApproximateTiming();
    }
}
