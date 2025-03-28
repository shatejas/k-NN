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
#include <jni.h>

#include <vector>

#include "jni_util.h"
#include "commons.h"

jlong knn_jni::commons::storeVectorData(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong memoryAddressJ,
                                        jobjectArray dataJ, jlong initialCapacityJ, jboolean appendJ) {
    std::vector<float> *vect;
    if ((long) memoryAddressJ == 0) {
        vect = new std::vector<float>();
        vect->reserve((long)initialCapacityJ);
    } else {
        vect = reinterpret_cast<std::vector<float>*>(memoryAddressJ);
    }

    if (appendJ == JNI_FALSE) {
        vect->clear();
    }

    int dim = jniUtil->GetInnerDimensionOf2dJavaFloatArray(env, dataJ);
    jniUtil->Convert2dJavaObjectArrayAndStoreToFloatVector(env, dataJ, dim, vect);

    return (jlong) vect;
}

jlong knn_jni::commons::storeBinaryVectorData(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong memoryAddressJ,
                                        jobjectArray dataJ, jlong initialCapacityJ, jboolean appendJ) {
    std::vector<uint8_t> *vect;
    if ((long) memoryAddressJ == 0) {
        vect = new std::vector<uint8_t>();
        vect->reserve((long)initialCapacityJ);
    } else {
        vect = reinterpret_cast<std::vector<uint8_t>*>(memoryAddressJ);
    }

    if (appendJ == JNI_FALSE) {
        vect->clear();
    }

    int dim = jniUtil->GetInnerDimensionOf2dJavaByteArray(env, dataJ);
    jniUtil->Convert2dJavaObjectArrayAndStoreToBinaryVector(env, dataJ, dim, vect);

    return (jlong) vect;
}

jlong knn_jni::commons::storeByteVectorData(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong memoryAddressJ,
                                        jobjectArray dataJ, jlong initialCapacityJ, jboolean appendJ) {
    std::vector<int8_t> *vect;
    if (memoryAddressJ == 0) {
        vect = new std::vector<int8_t>();
        vect->reserve(static_cast<long>(initialCapacityJ));
    } else {
        vect = reinterpret_cast<std::vector<int8_t>*>(memoryAddressJ);
    }

    if (appendJ == JNI_FALSE) {
            vect->clear();
    }

    int dim = jniUtil->GetInnerDimensionOf2dJavaByteArray(env, dataJ);
    jniUtil->Convert2dJavaObjectArrayAndStoreToByteVector(env, dataJ, dim, vect);

    return (jlong) vect;
}

void knn_jni::commons::freeVectorData(jlong memoryAddressJ) {
    if (memoryAddressJ != 0) {
        auto *vect = reinterpret_cast<std::vector<float>*>(memoryAddressJ);
        delete vect;
    }
}

void knn_jni::commons::freeBinaryVectorData(jlong memoryAddressJ) {
    if (memoryAddressJ != 0) {
        auto *vect = reinterpret_cast<std::vector<uint8_t>*>(memoryAddressJ);
        delete vect;
    }
}

void knn_jni::commons::freeByteVectorData(jlong memoryAddressJ) {
    if (memoryAddressJ != 0) {
        auto *vect = reinterpret_cast<std::vector<int8_t>*>(memoryAddressJ);
        delete vect;
    }
}

int knn_jni::commons::getIntegerMethodParameter(JNIEnv * env, knn_jni::JNIUtilInterface * jniUtil, std::unordered_map<std::string, jobject> methodParams, std::string methodParam, int defaultValue) {
    if (methodParams.empty()) {
        return defaultValue;
    }
    auto efSearchIt = methodParams.find(methodParam);
    if (efSearchIt != methodParams.end()) {
        return jniUtil->ConvertJavaObjectToCppInteger(env, methodParams[methodParam]);
    }

    return defaultValue;
}
