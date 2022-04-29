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


#include <filesystem>
#include <istream>
#include <string>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/numbers.h>
#include <absl/strings/substitute.h>
#include <fstream>

#include "src/stirling/source_connectors/mlexray_stats/mlexray_stats_parser.h"


constexpr int mlexrayLogNumFields = 2;

namespace px {
namespace stirling {

    Status MLEXrayStatsParser::parse_log(MLEXrayStats *out) const {
        /**
         * Directory:
         * Invocation
         * - Modelname
         * - - run_name
         * - - - dataset
         * - - - - log
         * - - - - nativeLog
         * - - - - summary.log
         */
        DCHECK(out != nullptr);
        Status log_parser_status;
//        Status native_log_parser_status;

        for (const auto & invocation : std::filesystem::directory_iterator(MLEXrayStatsParser::trace_path)){
            std::cout << invocation.path() << std::endl;
            if (!invocation.is_directory()) continue;
            for (const auto & model_name : std::filesystem::directory_iterator(invocation.path())){
                out->context.model_name = model_name.path();
                if (!model_name.is_directory()) continue;
                for (const auto & run_name : std::filesystem::directory_iterator(model_name.path())){
                    out->context.hw_descriptor = run_name.path();
                    if (!run_name.is_directory()) continue;
                    for (const auto & dataset_name : std::filesystem::directory_iterator(run_name.path())){
                        out->context.dataset = dataset_name.path();
                        log_parser_status = MLEXrayStatsParser::parse_mlexray_log(out, dataset_name.path().string() + "/log/");
//                        native_log_parser_status = MLEXrayStatsParser::parse_mlexray_log(out, dataset_name.path().string() + "/nativeLog/");
                        break;
                    }
                }
            }
        }

        return Status::OK();
    }

    Status MLEXrayStatsParser::parse_mlexray_log(MLEXrayStats *out, std::string log_path) const {
        /**
        * Sample log file:
        * Inference Start Time: 508833925 ms
        * Inference Time: 39551 ms
        * Inference Result: [34.0, 0.5529412, 65.0, 0.078431375, 33.0, 0.043137256]
        *
        */
        DCHECK(out != nullptr);
        for (const auto & frame : std::filesystem::directory_iterator(log_path)){
            // exclude meta.log
            std::size_t found = frame.path().string().find("meta.log");
            if (found != std::string::npos) continue;
            std::ifstream ifs;
            ifs.open(frame.path());
            if (!ifs) {
                return error::Internal("Failed to open file $0", frame.path().string());
            }

            // parse each frame into a span
            MLEXrayStatsParser::LayerSpan span = MLEXrayStatsParser::LayerSpan();
            std::string line;
            while (std::getline(ifs, line)) {
                std::vector<std::string> split = absl::StrSplit(line, ':', absl::SkipWhitespace());
                std::string key_name = split[0];
                std::string value = "";
                if (split.size()>=mlexrayLogNumFields){
                    value = split[1];
                }
                // parse all keys
                found = key_name.find("Inference Time");
                if (found != std::string::npos){
                    span.invocation_time_ns = value;
                    span.span_start = 0;
                    span.span_end = -1;
                }

                found = key_name.find("Inference Result");
                if (found != std::string::npos){
                    span.span_feature = value;
                }
            }
            out->layer_spans.push_back(span);

        }

        return Status::OK();
    }


//    Status MLEXrayStatsParser::parse_mlexray_native_log(MLEXrayStats *out, std::string log_path) const {
//        /**
//        * Sample native log file:
//        * Inference Start time: 1.63399e+12ms
//        * Inference Latency: 21721.7ms
//        * Inference memory usage:
//        * max resident set size = 2.36719 MB, total malloc-ed size = 0 MB, in-use allocated/mmapped size = -0.453033 MB
//        * Embeddings Dims: [ ]
//        * Embeddings:
//        * Layer Outputs:
//        * input_1: -79 -69 -67 -77 -67 -65 -75 -65 -63 -72 -62 -60 -71 -61 -60 -70 -60 -59 -69 -59 -58 -68 -58 -57 -68 -58 -57 -66 -57 -58 -66 -57 -58 -66 -57 -56 -65 -57 -55 -65 -57 -55 -65 -56 -56 -64 -56 -54 -63 -55 -53 -63 -54 -53 -64 -54 -52 -64 -56 -53 -64 -56 -54 -64 -55 -53 -63 -53 -52 -63 -53 -52 -63 -53 -52 -63 -53 ...
//        *
//        */
//        DCHECK(out != nullptr);
//
//        return Status::OK();
//    }

}
}