// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "faiss_index_service.h"
#include "faiss_methods.h"
#include "faiss/index_factory.h"
#include "faiss/Index.h"
#include "faiss/IndexBinary.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexBinaryHNSW.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexIDMap.h"
#include "faiss/index_io.h"
#include <algorithm>
#include "commons.h"
#include "faiss_util.h"
#include <string>
#include <vector>
#include <memory>
#include <type_traits>

namespace faiss {
    // Using jlong to do Bitmap selector, jlong[] equals to lucene FixedBitSet#bits
    struct IDSelectorBits : IDSelector {
        knn_jni::JNIUtilInterface *jni_util;
        JNIEnv *env;
        jclass clazz;
        jmethodID getMethodId;
        jobject acceptedDocs;

        /** Construct with a binary mask like Lucene FixedBitSet
         *
         * @param n size of the bitmap array
         * @param bitmap id like Lucene FixedBitSet bits
         */
        IDSelectorBits(JNIEnv *jni_env, knn_jni::JNIUtilInterface *jni_util, jobject acceptedDocs) : env(jni_env),
            jni_util(jni_util),
            clazz(jni_util->FindClass(
                jni_env, "Lorg/apache/lucene/util/Bits;")),
            getMethodId(
                jni_util->FindMethod(
                    jni_env, "Lorg/apache/lucene/util/Bits;",
                    "get")),
            acceptedDocs(acceptedDocs) {};

        bool is_member(idx_t id) const final {
            return env->CallBooleanMethod(acceptedDocs, getMethodId, id);
        }

        ~IDSelectorBits() override {
        }
    };
}


namespace knn_jni {
    namespace faiss_wrapper {
        template<typename INDEX, typename IVF, typename HNSW>
        void SetExtraParameters(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env,
                                const std::unordered_map<std::string, jobject> &parametersCpp, INDEX *index) {
            std::unordered_map<std::string, jobject>::const_iterator value;
            if (auto *indexIvf = dynamic_cast<IVF *>(index)) {
                if ((value = parametersCpp.find(knn_jni::NPROBES)) != parametersCpp.end()) {
                    indexIvf->nprobe = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
                }

                if ((value = parametersCpp.find(knn_jni::COARSE_QUANTIZER)) != parametersCpp.end()
                    && indexIvf->quantizer != nullptr) {
                    auto subParametersCpp = jniUtil->ConvertJavaMapToCppMap(env, value->second);
                    SetExtraParameters<INDEX, IVF, HNSW>(jniUtil, env, subParametersCpp, indexIvf->quantizer);
                }
            }

            if (auto *indexHnsw = dynamic_cast<HNSW *>(index)) {
                if ((value = parametersCpp.find(knn_jni::EF_CONSTRUCTION)) != parametersCpp.end()) {
                    indexHnsw->hnsw.efConstruction = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
                }

                if ((value = parametersCpp.find(knn_jni::EF_SEARCH)) != parametersCpp.end()) {
                    indexHnsw->hnsw.efSearch = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
                }
            }
        }

        IndexService::IndexService(std::unique_ptr<FaissMethods> faissMethods) : faissMethods(std::move(faissMethods)) {
        }

        void IndexService::createIndex(
            knn_jni::JNIUtilInterface *jniUtil,
            JNIEnv *env,
            faiss::MetricType metric,
            std::string indexDescription,
            int dim,
            int numIds,
            int threadCount,
            int64_t vectorsAddress,
            std::vector<int64_t> ids,
            std::string indexPath,
            std::unordered_map<std::string, jobject> parameters
        ) {
            // Read vectors from memory address
            auto *inputVectors = reinterpret_cast<std::vector<float> *>(vectorsAddress);

            // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
            int numVectors = (int) (inputVectors->size() / (uint64_t) dim);
            if (numVectors == 0) {
                throw std::runtime_error("Number of vectors cannot be 0");
            }

            if (numIds != numVectors) {
                throw std::runtime_error("Number of IDs does not match number of vectors");
            }

            std::unique_ptr<faiss::Index>
                    indexWriter(faissMethods->indexFactory(dim, indexDescription.c_str(), metric));

            // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
            if (threadCount != 0) {
                omp_set_num_threads(threadCount);
            }

            // Add extra parameters that cant be configured with the index factory
            SetExtraParameters<faiss::Index, faiss::IndexIVF, faiss::IndexHNSW>(
                jniUtil, env, parameters, indexWriter.get());

            // Check that the index does not need to be trained
            if (!indexWriter->is_trained) {
                throw std::runtime_error("Index is not trained");
            }

            // Add vectors
            std::unique_ptr<faiss::IndexIDMap> idMap(faissMethods->indexIdMap(indexWriter.get()));
            idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());

            // Write the index to disk
            faissMethods->writeIndex(idMap.get(), indexPath.c_str());
        }

        jobjectArray IndexService::searchIndex(
            JNIUtilInterface *jniUtil,
            JNIEnv *env,
            jlong indexPointerJ,
            jfloatArray queryVectorJ,
            jint kJ,
            jobject methodParamsJ,
            jobject acceptedDocs,
            jintArray parentIdsJ) {
            auto *indexReader = reinterpret_cast<faiss::IndexIDMap *>(indexPointerJ);
            std::unordered_map<std::string, jobject> methodParams;

            if (methodParamsJ != nullptr) {
                methodParams = jniUtil->ConvertJavaMapToCppMap(env, methodParamsJ);
            }

            std::vector<float> dis(kJ);
            std::vector<faiss::idx_t> ids(kJ);
            float *rawQueryvector = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr);
            /*
                Setting the omp_set_num_threads to 1 to make sure that no new OMP threads are getting created.
            */
            omp_set_num_threads(1);
            std::unique_ptr<faiss::IDSelector> idSelector;

            faiss::SearchParameters *searchParameters;
            faiss::SearchParametersHNSW hnswParams;
            std::unique_ptr<faiss::IDGrouperBitmap> idGrouper;
            std::vector<uint64_t> idGrouperBitmap;
            auto hnswReader = dynamic_cast<const faiss::IndexHNSW*>(indexReader->index);
            if(hnswReader) {
                // Query param efsearch supersedes ef_search provided during index setting.
                hnswParams.efSearch = knn_jni::commons::getIntegerMethodParameter(env, jniUtil, methodParams, EF_SEARCH, hnswReader->hnsw.efSearch);
                if (acceptedDocs != nullptr) {
                    idSelector.reset(new faiss::IDSelectorBits(env, jniUtil, acceptedDocs));
                    hnswParams.sel = idSelector.get();
                }
                searchParameters = &hnswParams;
            }

            try {
                indexReader->search(1, rawQueryvector, kJ, dis.data(), ids.data(), searchParameters);
            } catch (...) {
                jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);
                throw;
            }

            int resultSize = kJ;
            auto it = std::find(ids.begin(), ids.end(), -1);
            if (it != ids.end()) {
                resultSize = it - ids.begin();
            }

            jclass resultClass = jniUtil->FindClass(env,"org/opensearch/knn/index/query/KNNQueryResult");
            jmethodID allArgs = jniUtil->FindMethod(env, "org/opensearch/knn/index/query/KNNQueryResult", "<init>");

            jobjectArray results = jniUtil->NewObjectArray(env, resultSize, resultClass, nullptr);

            jobject result;
            for(int i = 0; i < resultSize; ++i) {
                result = jniUtil->NewObject(env, resultClass, allArgs, ids[i], dis[i]);
                jniUtil->SetObjectArrayElement(env, results, i, result);
            }
            return results;
        }

        BinaryIndexService::BinaryIndexService(std::unique_ptr<FaissMethods> faissMethods) : IndexService(
            std::move(faissMethods)) {
        }

        void BinaryIndexService::createIndex(
            knn_jni::JNIUtilInterface *jniUtil,
            JNIEnv *env,
            faiss::MetricType metric,
            std::string indexDescription,
            int dim,
            int numIds,
            int threadCount,
            int64_t vectorsAddress,
            std::vector<int64_t> ids,
            std::string indexPath,
            std::unordered_map<std::string, jobject> parameters
        ) {
            // Read vectors from memory address
            auto *inputVectors = reinterpret_cast<std::vector<uint8_t> *>(vectorsAddress);

            if (dim % 8 != 0) {
                throw std::runtime_error("Dimensions should be multiply of 8");
            }
            // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
            int numVectors = (int) (inputVectors->size() / (uint64_t) (dim / 8));
            if (numVectors == 0) {
                throw std::runtime_error("Number of vectors cannot be 0");
            }

            if (numIds != numVectors) {
                throw std::runtime_error("Number of IDs does not match number of vectors");
            }

            std::unique_ptr<faiss::IndexBinary> indexWriter(
                faissMethods->indexBinaryFactory(dim, indexDescription.c_str()));

            // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
            if (threadCount != 0) {
                omp_set_num_threads(threadCount);
            }

            // Add extra parameters that cant be configured with the index factory
            SetExtraParameters<faiss::IndexBinary, faiss::IndexBinaryIVF, faiss::IndexBinaryHNSW>(
                jniUtil, env, parameters, indexWriter.get());

            // Check that the index does not need to be trained
            if (!indexWriter->is_trained) {
                throw std::runtime_error("Index is not trained");
            }

            // Add vectors
            std::unique_ptr<faiss::IndexBinaryIDMap> idMap(faissMethods->indexBinaryIdMap(indexWriter.get()));
            idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());

            // Write the index to disk
            faissMethods->writeIndexBinary(idMap.get(), indexPath.c_str());
        }
    } // namespace faiss_wrapper
} // namesapce knn_jni
