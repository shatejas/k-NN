/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats.suppliers;

import lombok.RequiredArgsConstructor;
import org.opensearch.knn.plugin.stats.KNNTimer;

import java.util.function.Supplier;

@RequiredArgsConstructor
public class KNNTimerSupplier implements Supplier<Long> {

    private final KNNTimer knnTimer;

    @Override
    public Long get() {
        return knnTimer.average();
    }
}
