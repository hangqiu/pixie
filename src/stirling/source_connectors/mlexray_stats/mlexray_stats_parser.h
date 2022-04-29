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

#include <filesystem>
#include <istream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>


#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include "src/common/base/base.h"
#include "src/common/system/system.h"

namespace px {
namespace stirling {
/*
 * MLEXrayStatsParser is used to parse ml proc pseudo filesystem.
 */

class MLEXrayStatsParser{
public:
    MLEXrayStatsParser() = default;
    ~MLEXrayStatsParser() = default;
    const std::string trace_path = "/trace/";
    struct LayerSpan{
        int64_t span_start = 0;
        int64_t span_end = 0;
        std::string invocation_time_ns = ""; //TODO: to be changed to int64
        std::string span_feature;
//        std::vector<float> span_feature; //  TODO: uncomment
    };

    struct MLEXrayContext{
        int64_t context_id = 0;
        std::string model_name = "";
        std::string model_mode = ""; // eg. qunatized
        std::string hw_descriptor = ""; //eg. cloud, phone, arm, x86
        std::string dataset = "";
    };

    struct MLEXrayStats{
        int64_t invocation_id = 0;
        MLEXrayContext context = MLEXrayContext();
        std::vector<LayerSpan> layer_spans;

        void Clear() {
            *this = MLEXrayStats();
        }
    };

    Status parse_log(MLEXrayStats* out) const;
    Status parse_mlexray_log(MLEXrayStats* out, std::string log_path) const;
//    Status parse_mlexray_native_log(MLEXrayStats *out, std::string log_path) const;

};

}
}