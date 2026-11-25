# Full results (synthetic, 1,500 held-out scenes; mean over seeds where 2 exist)

| Configuration | seeds | occ mIoU <=12.8 m | occ mIoU full | sem mIoU (6 cls) | dyn macro acc | car | truck | cyclist | ped | vel MAE (m/s) | aliased | unaliased | candidate acc | static vel MAE | ego MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| imaging radar + external ego (baseline) | 2 | 0.796 | 0.796 | 0.545 | 0.760 | 0.851 | 0.804 | 0.734 | 0.648 | 4.007 | 4.237 | 3.373 | 0.427 | 0.143 | 0.177 |
| baseline, odometry noise 0.5 m/s | 2 | 0.796 | 0.796 | 0.545 | 0.759 | 0.852 | 0.804 | 0.733 | 0.647 | 4.110 | 4.337 | 3.482 | 0.426 | 0.408 | 0.177 |
| imaging + single-chip, radar ego (RadarOSV-style) | 2 | 0.802 | 0.797 | 0.547 | 0.767 | 0.840 | 0.793 | 0.739 | 0.694 | 0.592 | 0.599 | 0.573 | 0.937 | 0.228 | 0.134 |
| imaging + single-chip, external ego | 1 | 0.805 | 0.802 | 0.554 | 0.772 | 0.839 | 0.799 | 0.739 | 0.713 | 0.506 | 0.515 | 0.478 | 0.940 | 0.166 | 0.133 |
| RadarOSV-style, fixed-only PCG | 1 | 0.805 | 0.801 | 0.554 | 0.782 | 0.831 | 0.793 | 0.772 | 0.733 | 0.584 | 0.598 | 0.546 | 0.939 | 0.236 | 0.139 |

Share of dynamic cells aliased in the imaging radar: 0.734

## Per-seed

| Configuration | seed | occ mIoU <=12.8 m | dyn macro acc | vel MAE | ego MAE |
|---|---|---|---|---|---|
| base | 0 | 0.797 | 0.763 | 4.062 | 0.184 |
| base | 1 | 0.794 | 0.756 | 3.952 | 0.170 |
| osv | 0 | 0.798 | 0.770 | 0.572 | 0.131 |
| osv | 1 | 0.806 | 0.763 | 0.612 | 0.137 |
| osv_ext | 0 | 0.805 | 0.772 | 0.506 | 0.133 |
| osv_fixed | 0 | 0.805 | 0.782 | 0.584 | 0.139 |
| base_odo | 0 | 0.797 | 0.763 | 4.163 | 0.184 |
| base_odo | 1 | 0.794 | 0.755 | 4.057 | 0.170 |

## Paired differences (seed 0), 95% CI by scene bootstrap (500 resamples)

| Comparison | diff | CI |
|---|---|---|
| osv_minus_base:vel_mae_all:seed0 | -3.490 | -3.602 .. -3.376 |
| osv_minus_base:dyn_macro_acc:seed0 | +0.007 | -0.000 .. +0.014 |
| osv_minus_base:occ_miou_near:seed0 | +0.000 | -0.002 .. +0.002 |
| osv_minus_base:occ_miou_full:seed0 | -0.003 | -0.005 .. -0.002 |
| osv_minus_base:cand_acc:seed0 | +0.515 | +0.502 .. +0.528 |
| osv_ext_minus_osv:vel_mae_all:seed0 | -0.066 | -0.086 .. -0.047 |
| osv_ext_minus_osv:dyn_macro_acc:seed0 | +0.002 | -0.003 .. +0.007 |
| osv_ext_minus_osv:occ_miou_near:seed0 | +0.008 | +0.006 .. +0.009 |
| osv_ext_minus_osv:occ_miou_full:seed0 | +0.007 | +0.006 .. +0.008 |
| osv_ext_minus_osv:cand_acc:seed0 | +0.001 | -0.001 .. +0.003 |
| osv_minus_osv_fixed:vel_mae_all:seed0 | -0.012 | -0.031 .. +0.008 |
| osv_minus_osv_fixed:dyn_macro_acc:seed0 | -0.012 | -0.018 .. -0.006 |
| osv_minus_osv_fixed:occ_miou_near:seed0 | -0.008 | -0.010 .. -0.006 |
| osv_minus_osv_fixed:occ_miou_full:seed0 | -0.006 | -0.007 .. -0.005 |
| osv_minus_osv_fixed:cand_acc:seed0 | -0.000 | -0.002 .. +0.002 |
| osv_minus_base_odo:vel_mae_all:seed0 | -3.591 | -3.698 .. -3.479 |
| osv_minus_base_odo:dyn_macro_acc:seed0 | +0.007 | +0.000 .. +0.014 |
| osv_minus_base_odo:occ_miou_near:seed0 | +0.000 | -0.002 .. +0.002 |
| osv_minus_base_odo:occ_miou_full:seed0 | -0.003 | -0.005 .. -0.002 |
| osv_minus_base_odo:cand_acc:seed0 | +0.515 | +0.501 .. +0.527 |
