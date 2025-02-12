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

#include "src/stirling/source_connectors/mlexray_stats/mlexray_stats_connector.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include "src/common/base/base.h"
#include "src/common/system/proc_parser.h"
#include "src/shared/metadata/metadata.h"

#include "src/stirling/source_connectors/mlexray_stats/mlexray_stats_parser.h"

namespace px {
namespace stirling {

//using system::ProcParser;

Status MLEXrayStatsConnector::InitImpl() {
  sampling_freq_mgr_.set_period(kSamplingPeriod);
  push_freq_mgr_.set_period(kPushPeriod);
  return Status::OK();
}

Status MLEXrayStatsConnector::StopImpl() { return Status::OK(); }

void MLEXrayStatsConnector::TransferMLEXrayStatsTable(ConnectorContext* ctx,
                                                      DataTable* data_table) {
  const absl::flat_hash_map<md::UPID, md::PIDInfoUPtr>& pid_info_by_upid = ctx->GetPIDInfoMap();

  int64_t timestamp = AdjustedSteadyClockNowNS();

  for (const auto& [upid, pid_info] : pid_info_by_upid) {
    // TODO(zasgar): Fix condition for dead pids after helper function is added.
    if (pid_info == nullptr || pid_info->stop_time_ns() > 0) {
      // PID has been stopped.
      continue;
    }

    MLEXrayStatsParser MLStatsParser = MLEXrayStatsParser();
    MLEXrayStatsParser::MLEXrayStats MLStats = MLEXrayStatsParser::MLEXrayStats();
    auto s = MLStatsParser.parse_log(&MLStats);

    if (!s.ok()) {
        VLOG(1) << absl::StrCat("Failed to parse MLExray stats: ", s.msg());
        continue;
    }

    for (MLEXrayStatsParser::LayerSpan span : MLStats.layer_spans){
        DataTable::RecordBuilder<&kMLEXrayStatsTable> r(data_table, timestamp);
        // TODO(oazizi): Enable version below, once rest of the agent supports tabletization.
        //  DataTable::RecordBuilder<&kMLEXrayStatsTable> r(data_table, upid.value(), timestamp);
        r.Append<r.ColIndex("time_")>(timestamp);
        // Tabletization key must also be appended as a column value.
        // See note in RecordBuilder class.
        r.Append<r.ColIndex("invocation_id")>(0);
        r.Append<r.ColIndex("invocation_time_ns")>(span.invocation_time_ns);
        r.Append<r.ColIndex("span_start")>(span.span_start);
        r.Append<r.ColIndex("span_end")>(span.span_end);
        r.Append<r.ColIndex("span_feature")>(span.span_feature);
        r.Append<r.ColIndex("context_id")>(0);
    }
  }
}

void MLEXrayStatsConnector::TransferDataImpl(ConnectorContext* ctx,
                                             const std::vector<DataTable*>& data_tables) {
  DCHECK_EQ(data_tables.size(), 1);

  auto* data_table = data_tables[0];

  if (data_table == nullptr) {
    return;
  }

  TransferMLEXrayStatsTable(ctx, data_table);
}

}  // namespace stirling
}  // namespace px
