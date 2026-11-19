# Results (synthetic toy, 3600 held-out windows from 30 unseen drivers)

Mean +- sd over seeds [0, 1, 2]. Paired AUC = location-matched impaired vs baseline window.

## Impairment inference

| Labels | Variant | 4-way acc | macro-F1 | AUC (impaired vs none) | paired AUC | cog | visual | emotional | visual-vs-cognitive AUC |
|---|---|---|---|---|---|---|---|---|---|
| 100% | idmae | 0.646 +- 0.002 | 0.656 +- 0.004 | 0.781 +- 0.005 | 0.808 +- 0.006 | 0.779 | 0.826 | 0.819 | 0.884 +- 0.002 |
| 100% | norecon | 0.621 +- 0.005 | 0.633 +- 0.006 | 0.761 +- 0.006 | 0.800 +- 0.014 | 0.729 | 0.836 | 0.836 | 0.869 +- 0.004 |
| 100% | gru | 0.652 +- 0.005 | 0.664 +- 0.005 | 0.789 +- 0.006 | 0.800 +- 0.009 | 0.757 | 0.803 | 0.842 | 0.893 +- 0.007 |
| 10% | idmae | 0.623 +- 0.002 | 0.628 +- 0.003 | 0.765 +- 0.001 | 0.788 +- 0.001 | 0.760 | 0.803 | 0.800 | 0.887 +- 0.005 |
| 10% | norecon | 0.632 +- 0.010 | 0.641 +- 0.008 | 0.769 +- 0.006 | 0.798 +- 0.008 | 0.769 | 0.829 | 0.796 | 0.883 +- 0.012 |
| 10% | gru | 0.610 +- 0.005 | 0.616 +- 0.006 | 0.743 +- 0.004 | 0.756 +- 0.005 | 0.700 | 0.799 | 0.771 | 0.875 +- 0.005 |

## Q1 / Q2 - paired-AUC differences (seed-ensemble scores, 95% CI by location bootstrap)

* idmae_minus_norecon_f1.0: +0.012 (CI -0.000 .. +0.023)
* idmae_minus_gru_f1.0: +0.016 (CI -0.000 .. +0.033)
* idmae_minus_norecon_f0.1: -0.009 (CI -0.023 .. +0.005)
* idmae_minus_gru_f0.1: +0.047 (CI +0.027 .. +0.069)

## Paired AUC by driver susceptibility (low < 0.4 <= mid < 0.7 <= high)

| Labels | Variant | low | mid | high |
|---|---|---|---|---|
| 100% | idmae | 0.706 | 0.849 | 0.927 |
| 100% | norecon | 0.708 | 0.843 | 0.904 |
| 100% | gru | 0.694 | 0.845 | 0.923 |
| 10% | idmae | 0.687 | 0.822 | 0.913 |
| 10% | norecon | 0.694 | 0.840 | 0.919 |
| 10% | gru | 0.645 | 0.806 | 0.883 |

## Q3 - linear probe on the label-token embedding: visual vs cognitive AUC

* idmae_f1.0: 0.903 +- 0.003
* norecon_f1.0: 0.889 +- 0.006
* idmae_f0.1: 0.901 +- 0.002
* norecon_f0.1: 0.902 +- 0.010

## Q4 - forecasting the last 2 s (ID-MAE trained with 100% labels; metrics averaged over seeds)

| Method | position ADE (m) | position FDE (m) | lateral ADE (m) | speed RMSE (m/s) | gaze error (deg) |
|---|---|---|---|---|---|
| kinematic | 0.227 | 0.429 | 0.211 | 0.070 | 12.76 |
| idmae_iterative | 0.530 | 0.965 | 0.198 | 0.516 | 9.18 |
| idmae_oneshot | 0.483 | 0.812 | 0.199 | 0.431 | 9.18 |
| idmae_iterative_true_label | 0.527 | 0.960 | 0.198 | 0.513 | 9.19 |

### Gaze error (deg) by condition

| Method | none | cognitive | visual | emotional |
|---|---|---|---|---|
| kinematic | 12.77 | 10.22 | 14.51 | 13.53 |
| idmae_iterative | 8.97 | 7.20 | 11.14 | 9.42 |
| idmae_oneshot | 8.97 | 7.20 | 11.14 | 9.42 |
| idmae_iterative_true_label | 8.97 | 7.20 | 11.15 | 9.42 |
