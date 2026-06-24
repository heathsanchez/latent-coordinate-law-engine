# Latent Coordinate Law Demo Summary

Rows analyzed: 600
Top t coordinate: support_face
Top r coordinate: threshold_zone
Induced laws: 26
Combined holdout accuracy: 1.000
Counterexamples found: 0

## Laws

- if support_index >= 13 then t = 25000 (precision=1.000, recall=1.000)
- if support_face == x13_x15_x17 then t = 25000 (precision=1.000, recall=1.000)
- if support_bucket == x13_x15_x17 then t = 25000 (precision=1.000, recall=1.000)
- if support_index <= 12 then t = 24979 (precision=1.000, recall=1.000)
- if support_index == 12 then t = 24979 (precision=1.000, recall=1.000)
- if active_support_slots == 12 then t = 24979 (precision=1.000, recall=1.000)
- if support_face == x12_base then t = 24979 (precision=1.000, recall=1.000)
- if support_bucket == x12_base then t = 24979 (precision=1.000, recall=1.000)
- if support_index >= 15 then t = 25000 (precision=1.000, recall=0.686)
- if a6 <= 70 then r = 2 (precision=1.000, recall=1.000)
- if threshold_zone == low then r = 2 (precision=1.000, recall=1.000)
- if a6 >= 72 then r = 6 (precision=1.000, recall=1.000)
