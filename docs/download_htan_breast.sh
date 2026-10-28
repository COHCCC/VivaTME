#!/bin/bash
# Download the HTAN breast Visium sections used in the paper from Synapse (see docs/download_htan_breast.md).
# Prerequisites: pip install synapseclient; synapse login (personal access token from synapse.org).
# usage: bash docs/download_htan_breast.sh OUT_DIR
set -euo pipefail
OUT=$(cd "${1:-.}" && pwd)
# ---- HTA12_129_9 (7 files) ----
mkdir -p "$OUT/HTA12_129_9"
cd "$OUT/HTA12_129_9"
synapse get syn53215089
synapse get syn53215160
synapse get syn53215176
synapse get syn53215259
synapse get syn53215269
synapse get syn53215307
synapse get syn53215842
cd "$OUT"
# ---- HTA12_134_7 (7 files) ----
mkdir -p "$OUT/HTA12_134_7"
cd "$OUT/HTA12_134_7"
synapse get syn53215088
synapse get syn53215094
synapse get syn53215147
synapse get syn53215205
synapse get syn53215270
synapse get syn53215293
synapse get syn53215864
cd "$OUT"
# ---- HTA12_135_10 (7 files) ----
mkdir -p "$OUT/HTA12_135_10"
cd "$OUT/HTA12_135_10"
synapse get syn53215122
synapse get syn53215124
synapse get syn53215164
synapse get syn53215226
synapse get syn53215227
synapse get syn53215248
synapse get syn53215850
cd "$OUT"
# ---- HTA12_139_3 (7 files) ----
mkdir -p "$OUT/HTA12_139_3"
cd "$OUT/HTA12_139_3"
synapse get syn53215141
synapse get syn53215143
synapse get syn53215200
synapse get syn53215224
synapse get syn53215260
synapse get syn53215287
synapse get syn53215847
cd "$OUT"
# ---- HTA12_140_5 (7 files) ----
mkdir -p "$OUT/HTA12_140_5"
cd "$OUT/HTA12_140_5"
synapse get syn53215114
synapse get syn53215137
synapse get syn53215151
synapse get syn53215156
synapse get syn53215204
synapse get syn53215279
synapse get syn53215843
cd "$OUT"
# ---- HTA12_142_9 (7 files) ----
mkdir -p "$OUT/HTA12_142_9"
cd "$OUT/HTA12_142_9"
synapse get syn53215086
synapse get syn53215099
synapse get syn53215120
synapse get syn53215194
synapse get syn53215207
synapse get syn53215286
synapse get syn53215845
cd "$OUT"
# ---- HTA12_144_5 (7 files) ----
mkdir -p "$OUT/HTA12_144_5"
cd "$OUT/HTA12_144_5"
synapse get syn53215150
synapse get syn53215170
synapse get syn53215171
synapse get syn53215178
synapse get syn53215219
synapse get syn53215275
synapse get syn53215844
cd "$OUT"
# ---- HTA12_145_4 (7 files) ----
mkdir -p "$OUT/HTA12_145_4"
cd "$OUT/HTA12_145_4"
synapse get syn53215152
synapse get syn53215182
synapse get syn53215229
synapse get syn53215256
synapse get syn53215261
synapse get syn53215262
synapse get syn53215867
cd "$OUT"
# ---- HTA12_246_3 (7 files) ----
mkdir -p "$OUT/HTA12_246_3"
cd "$OUT/HTA12_246_3"
synapse get syn53215095
synapse get syn53215126
synapse get syn53215172
synapse get syn53215189
synapse get syn53215274
synapse get syn53215289
synapse get syn53215832
cd "$OUT"
# ---- HTA12_247_5 (7 files) ----
mkdir -p "$OUT/HTA12_247_5"
cd "$OUT/HTA12_247_5"
synapse get syn61462147
synapse get syn61462251
synapse get syn61462320
synapse get syn61462385
synapse get syn61462405
synapse get syn61462529
synapse get syn61465466
cd "$OUT"
# ---- HTA12_248_8 (7 files) ----
mkdir -p "$OUT/HTA12_248_8"
cd "$OUT/HTA12_248_8"
synapse get syn53215090
synapse get syn53215105
synapse get syn53215110
synapse get syn53215206
synapse get syn53215223
synapse get syn53215265
synapse get syn53215852
cd "$OUT"
# ---- HTA12_249_7 (7 files) ----
mkdir -p "$OUT/HTA12_249_7"
cd "$OUT/HTA12_249_7"
synapse get syn53215128
synapse get syn53215162
synapse get syn53215216
synapse get syn53215235
synapse get syn53215252
synapse get syn53215282
synapse get syn53471082
cd "$OUT"
# ---- HTA12_250_7 (7 files) ----
mkdir -p "$OUT/HTA12_250_7"
cd "$OUT/HTA12_250_7"
synapse get syn53215154
synapse get syn53215177
synapse get syn53215202
synapse get syn53215214
synapse get syn53215230
synapse get syn53215254
synapse get syn53215857
cd "$OUT"
# ---- HTA12_251_4 (7 files) ----
mkdir -p "$OUT/HTA12_251_4"
cd "$OUT/HTA12_251_4"
synapse get syn53215096
synapse get syn53215109
synapse get syn53215131
synapse get syn53215186
synapse get syn53215208
synapse get syn53215245
synapse get syn53215840
cd "$OUT"
# ---- HTA12_253_9 (7 files) ----
mkdir -p "$OUT/HTA12_253_9"
cd "$OUT/HTA12_253_9"
synapse get syn61462019
synapse get syn61462133
synapse get syn61462180
synapse get syn61462216
synapse get syn61462278
synapse get syn61462467
synapse get syn61465364
cd "$OUT"
# ---- HTA12_254_5 (7 files) ----
mkdir -p "$OUT/HTA12_254_5"
cd "$OUT/HTA12_254_5"
synapse get syn61462053
synapse get syn61462153
synapse get syn61462173
synapse get syn61462310
synapse get syn61462337
synapse get syn61462444
synapse get syn61465337
cd "$OUT"
# ---- HTA12_283_1 (7 files) ----
mkdir -p "$OUT/HTA12_283_1"
cd "$OUT/HTA12_283_1"
synapse get syn61462205
synapse get syn61462208
synapse get syn61462210
synapse get syn61462350
synapse get syn61462431
synapse get syn61462453
synapse get syn61465381
cd "$OUT"
# ---- HTA12_285_1 (7 files) ----
mkdir -p "$OUT/HTA12_285_1"
cd "$OUT/HTA12_285_1"
synapse get syn61462035
synapse get syn61462102
synapse get syn61462114
synapse get syn61462188
synapse get syn61462274
synapse get syn61462468
synapse get syn61465476
cd "$OUT"
# ---- HTA12_266_5 (7 files) ----
mkdir -p "$OUT/HTA12_266_5"
cd "$OUT/HTA12_266_5"
synapse get syn61462281
synapse get syn61462303
synapse get syn61462323
synapse get syn61462366
synapse get syn61462384
synapse get syn61462388
synapse get syn61465385
cd "$OUT"
# ---- HTA12_271_5 (7 files) ----
mkdir -p "$OUT/HTA12_271_5"
cd "$OUT/HTA12_271_5"
synapse get syn61462143
synapse get syn61462247
synapse get syn61462330
synapse get syn61462333
synapse get syn61462372
synapse get syn61462433
synapse get syn61465324
cd "$OUT"
