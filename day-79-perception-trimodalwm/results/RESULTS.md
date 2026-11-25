# Full results (600 held-out clips, 100 per weather, 6 observed -> 8 forecast frames)

Errors at +2 s unless marked mean (= mean over the 8 forecast frames). RGB / thermal MAE in 0-255 grey levels; LiDAR MAE in metres against measured returns.

## Cameras

| Model | RGB MAE +2 s | RGB MAE mean | RGB PSNR mean | thermal MAE +2 s | thermal MAE mean |
|---|---|---|---|---|---|
| rgb | 7.74 | 7.61 | 25.81 | - | - |
| thermal | - | - | - | 4.27 | 3.89 |
| base | 15.07 | 12.76 | 22.23 | - | - |
| tri | 8.55 | 8.35 | 25.03 | 4.59 | 4.32 |
| persist_rgb | 9.81 | 9.53 | 22.74 | - | - |
| persist_thermal | - | - | - | 6.19 | 5.63 |

## LiDAR (metres)

| Model | all returns +2 s | all mean | common returns +2 s | object returns +2 s | object mean |
|---|---|---|---|---|---|
| lidar | 0.46 | 0.42 | 0.35 | 5.23 | 4.54 |
| tri | 0.46 | 0.42 | 0.36 | 5.25 | 4.61 |
| persist_lidar | 1.12 | 1.08 | 0.39 | 9.76 | 9.46 |

'common' = returns measured both in the target frame and in the last observed frame (where persistence is defined).

## Paired differences, 95% CI by clip bootstrap

| Comparison | diff | CI |
|---|---|---|
| tri_minus_lidar/lidar_all_2s | +0.005 | +0.004 .. +0.007 |
| tri_minus_lidar/lidar_obj_2s | +0.011 | -0.014 .. +0.036 |
| tri_minus_persist/lidar_common_2s | -0.030 | -0.036 .. -0.025 |
| lidar_minus_persist/lidar_common_2s | -0.034 | -0.039 .. -0.028 |
| tri_minus_rgb/rgb_mae_2s | +0.810 | +0.781 .. +0.841 |
| tri_minus_thermal/thermal_mae_2s | +0.323 | +0.302 .. +0.340 |
| rgb_minus_persist/rgb_mae_2s | -2.077 | -2.162 .. -1.992 |
| thermal_minus_persist/thermal_mae_2s | -1.922 | -2.015 .. -1.817 |

## By weather (+2 s)

| Model / metric | clear | dusk | night | rain | fog | snow |
|---|---|---|---|---|---|---|
| rgb rgb_mae_2s | 8.41 | 7.15 | 6.98 | 7.26 | 6.44 | 10.18 |
| tri rgb_mae_2s | 9.54 | 7.88 | 7.48 | 8.08 | 7.04 | 11.27 |
| persist_rgb rgb_mae_2s | 11.32 | 8.84 | 8.04 | 9.61 | 8.16 | 12.91 |
| thermal thermal_mae_2s | 4.93 | 4.37 | 3.76 | 3.93 | 3.75 | 4.87 |
| tri thermal_mae_2s | 5.33 | 4.73 | 4.07 | 4.24 | 3.98 | 5.19 |
| persist_thermal thermal_mae_2s | 7.20 | 6.73 | 5.86 | 5.80 | 5.09 | 6.47 |
| lidar lidar_all_2s | 0.43 | 0.40 | 0.36 | 0.42 | 0.33 | 0.88 |
| tri lidar_all_2s | 0.43 | 0.40 | 0.35 | 0.43 | 0.34 | 0.89 |
| lidar lidar_common_2s | 0.34 | 0.31 | 0.28 | 0.32 | 0.26 | 0.69 |
| tri lidar_common_2s | 0.34 | 0.31 | 0.28 | 0.32 | 0.27 | 0.71 |
| persist_lidar lidar_common_2s | 0.38 | 0.37 | 0.33 | 0.32 | 0.28 | 0.72 |
| lidar lidar_obj_2s | 5.29 | 5.58 | 5.50 | 4.91 | 4.49 | 5.48 |
| tri lidar_obj_2s | 5.28 | 5.57 | 5.38 | 4.94 | 4.54 | 5.59 |
| persist_lidar lidar_obj_2s | 10.13 | 11.44 | 11.86 | 9.31 | 7.48 | 8.07 |

## Blanking one sensor in the tri model: % change of the other sensors' +2 s error

| blanked | other | clear | dusk | night | rain | fog | snow |
|---|---|---|---|---|---|---|---|
| rgb | thermal | +3.9% | +2.7% | +2.1% | +4.6% | +6.8% | +2.9% |
| rgb | lidar | +1.0% | +0.7% | +1.2% | +2.6% | +1.7% | +3.2% |
| thermal | rgb | +2.5% | +3.1% | +0.3% | +3.3% | +7.5% | +3.8% |
| thermal | lidar | +3.1% | +3.6% | +4.8% | +5.4% | +5.1% | +1.0% |
| lidar | rgb | +25.3% | +25.1% | +20.7% | +54.7% | +50.0% | +15.0% |
| lidar | thermal | +89.3% | +115.9% | +170.8% | +130.8% | +102.7% | +45.7% |
