# Original-dataset results

The native score is PR-AUC for the classifier and Dice for the segmentor.
best_epoch in the JSON is zero-based. Loss charts label the one-based epoch.

## Positive rate

| split | positive_rate |
| --- | --- |
| train | 0.1802 |
| val | 0.2091 |
| test | 0.1281 |

## Native matrix

| task | subset | side_px | gsd_m | score |
| --- | --- | --- | --- | --- |
| classify | s2_12 | 120 | 10 | 0.8340 |
| classify | rgb | 120 | 10 | 0.6965 |
| classify | loo_B1 | 120 | 10 | 0.7573 |
| classify | loo_B2 | 120 | 10 | 0.8263 |
| classify | loo_B3 | 120 | 10 | 0.7503 |
| classify | loo_B4 | 120 | 10 | 0.7454 |
| classify | loo_B5 | 120 | 10 | 0.8085 |
| classify | loo_B6 | 120 | 10 | 0.8055 |
| classify | loo_B7 | 120 | 10 | 0.7187 |
| classify | loo_B8 | 120 | 10 | 0.7708 |
| classify | loo_B8A | 120 | 10 | 0.8303 |
| classify | loo_B9 | 120 | 10 | 0.7461 |
| classify | loo_B11 | 120 | 10 | 0.7668 |
| classify | loo_B12 | 120 | 10 | 0.7402 |
| segment | s2_12 | 120 | 10 | 0.7779 |
| segment | rgb | 120 | 10 | 0.7591 |
| segment | loo_B1 | 120 | 10 | 0.7993 |
| segment | loo_B2 | 120 | 10 | 0.7773 |
| segment | loo_B3 | 120 | 10 | 0.7814 |
| segment | loo_B4 | 120 | 10 | 0.7188 |
| segment | loo_B5 | 120 | 10 | 0.7743 |
| segment | loo_B6 | 120 | 10 | 0.7678 |
| segment | loo_B7 | 120 | 10 | 0.7338 |
| segment | loo_B8 | 120 | 10 | 0.7728 |
| segment | loo_B8A | 120 | 10 | 0.7444 |
| segment | loo_B9 | 120 | 10 | 0.7644 |
| segment | loo_B11 | 120 | 10 | 0.7612 |
| segment | loo_B12 | 120 | 10 | 0.7640 |

## Ground sample distance

| task | subset | side_px | gsd_m | score |
| --- | --- | --- | --- | --- |
| classify | s2_12 | 120 | 10 | 0.8340 |
| classify | s2_12 | 80 | 15 | 0.7878 |
| classify | s2_12 | 60 | 20 | 0.7341 |
| classify | s2_12 | 40 | 30 | 0.6805 |
| classify | s2_12 | 30 | 40 | 0.6664 |
| classify | rgb | 120 | 10 | 0.6965 |
| classify | rgb | 80 | 15 | 0.6729 |
| classify | rgb | 60 | 20 | 0.7284 |
| classify | rgb | 40 | 30 | 0.6188 |
| classify | rgb | 30 | 40 | 0.7078 |
| segment | s2_12 | 120 | 10 | 0.7779 |
| segment | s2_12 | 80 | 15 | 0.7223 |
| segment | s2_12 | 60 | 20 | 0.6800 |
| segment | s2_12 | 40 | 30 | 0.6367 |
| segment | s2_12 | 30 | 40 | 0.6121 |
| segment | rgb | 120 | 10 | 0.7591 |
| segment | rgb | 80 | 15 | 0.7421 |
| segment | rgb | 60 | 20 | 0.6687 |
| segment | rgb | 40 | 30 | 0.6462 |
| segment | rgb | 30 | 40 | 0.6282 |
