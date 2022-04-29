/*
 * Copyright 2018- The Pixie Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/common/base/base.h"
#include "src/common/system/system.h"
#include "src/shared/metadata/metadata.h"
#include "src/stirling/core/canonical_types.h"
#include "src/stirling/core/source_connector.h"

namespace px {
    namespace stirling {
        struct MLEXrayStats {
            int64_t invocation_id = 0;
            int64_t invocation_time_ns = 0;
            std::string span_start = "";
            std::string span_end = "";
            int64_t span_feature = 0;
            int64_t context_id = 0;

            void Clear() { *this = MLEXrayStats(); }
        };

// clang-format off
        static constexpr DataElement kMLEXrayStatsElements[] = {
                canonical_data_elements::kTime,
//                canonical_data_elements::kUPID,
                {"invocation_id", "Model Invocation ID",
                 types::DataType::INT64, types::SemanticType::ST_NONE, types::PatternType::METRIC_COUNTER},
//                {"invocation_time_ns", "Cumulative time spent per model invocation",
//                 types::DataType::INT64, types::SemanticType::ST_DURATION_NS,
//                 types::PatternType::METRIC_GAUGE},
                {"invocation_time_ns", "Cumulative time spent per model invocation",
                 types::DataType::STRING, types::SemanticType::ST_NONE,
                 types::PatternType::METRIC_GAUGE},
                {"span_start", "Starting layer of model",
                 types::DataType::INT64, types::SemanticType::ST_NONE,
                 types::PatternType::METRIC_COUNTER},
                {"span_end", "Ending layer of model",
                 types::DataType::INT64, types::SemanticType::ST_NONE,
                 types::PatternType::METRIC_COUNTER},
                {"span_feature", "Summarized output of the span",
                 types::DataType::STRING, types::SemanticType::ST_NONE, types::PatternType::METRIC_GAUGE},
                {"context_id", "Model invocation context ID",
                 types::DataType::INT64, types::SemanticType::ST_NONE, types::PatternType::METRIC_COUNTER},
        };

        constexpr DataTableSchema kMLEXrayStatsTable(
                "mlexray_stats",
                "ML Model Invocation Summary",
                kMLEXrayStatsElements
        );
// clang-format on
        DEFINE_PRINT_TABLE(MLEXrayStats)

    }  // namespace stirling
}  // namespace px
