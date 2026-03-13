# haps

HAPS マルチビーム通信の Python シミュレーションコードです。  
ビーム半径、ビーム間隔、周波数再利用、災害時限定照射が通信性能に与える影響を評価します。

## Overview

- 対象: HAPS 下りリンク
- 評価指標: システムスループット、5%-tile SINR、SINR 標準偏差
- シナリオ（沖縄）:
  - 平常時の広域サービス
  - 災害時の限定領域サービス

## Repository

```text
src/
├─ beam_arrangement_haps*.py
├─ simulation_bandwidth_mp2_haps*.py
├─ database/
└─ japan*.json / geojson

ファイル名のdはdisaster、sはsingleを指す
