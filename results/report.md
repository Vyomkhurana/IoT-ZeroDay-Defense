# PPFL-IoT-ZeroDay — experiment report
_Generated 2026-10-28T17:26:01+00:00 from `results/experiments/*/metrics.json`._

## Detection performance
Unseen-attack recall = fraction of held-out (zero-day-like) attack samples flagged as anomalous.

| Experiment | Data | **Unseen recall** | Known recall | F1 | Precision | Recall | FPR | Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Centralized | nbaiot | 0.9990 | 0.9995 | 0.9935 | 0.9880 | 0.9991 | 0.0520 | 0.9894 | 0.9932 |
| Isolation Forest (centralized) | nbaiot | 0.5943 | 0.0028 | 0.6161 | 0.9747 | 0.4504 | 0.0500 | 0.5449 | 0.9415 |
| One-Class SVM (centralized) | nbaiot | 0.8042 | 0.4588 | 0.8317 | 0.9841 | 0.7202 | 0.0499 | 0.7637 | 0.9578 |
| FL + DP | nbaiot | 0.6717 | 0.2662 | 0.7231 | 0.9794 | 0.5731 | 0.0516 | 0.6441 | 0.9371 |
| Plain FL (FedAvg) [fedavg] | nbaiot | 0.9978 | 0.9971 | 0.9928 | 0.9880 | 0.9976 | 0.0521 | 0.9882 | 0.9891 |
| Plain FL (FedAvg) [fedavg_alpha_0.1] | nbaiot | 0.9965 | 0.9908 | 0.9919 | 0.9888 | 0.9951 | 0.0484 | 0.9869 | 0.9874 |
| Plain FL (FedAvg) [fedavg_alpha_0.3] | nbaiot | 0.9938 | 0.5705 | 0.9362 | 0.9866 | 0.8908 | 0.0520 | 0.9016 | 0.9828 |
| Plain FL (FedAvg) [fedavg_alpha_1.0] | nbaiot | 0.9973 | 0.9908 | 0.9921 | 0.9886 | 0.9957 | 0.0493 | 0.9872 | 0.9879 |
| Plain FL (FedAvg) [fedavg_alpha_10.0] | nbaiot | 0.9976 | 0.9926 | 0.9924 | 0.9884 | 0.9964 | 0.0501 | 0.9876 | 0.9883 |
| FedProx | nbaiot | 0.9948 | 0.9902 | 0.9908 | 0.9879 | 0.9937 | 0.0521 | 0.9850 | 0.9861 |
| Full PPFL (FedAvg) [full_ppfl] | nbaiot | 0.6717 | 0.2662 | 0.7231 | 0.9794 | 0.5731 | 0.0516 | 0.6441 | 0.9371 |
| Full PPFL (FedProx) | nbaiot | 0.6440 | 0.1790 | 0.6882 | 0.9780 | 0.5309 | 0.0512 | 0.6100 | 0.9337 |
| Full PPFL (FedAvg) [full_ppfl_tuned] | nbaiot | 0.8516 | 0.4694 | 0.8568 | 0.9843 | 0.7586 | 0.0520 | 0.7944 | 0.9651 |
| Full PPFL (FedAvg) [ppfl_clients_10] | nbaiot | 0.6717 | 0.2662 | 0.7231 | 0.9794 | 0.5731 | 0.0516 | 0.6441 | 0.9371 |
| Full PPFL (FedAvg) [ppfl_clients_20] | nbaiot | 0.4518 | 0.0037 | 0.5062 | 0.9675 | 0.3428 | 0.0494 | 0.4578 | 0.9154 |
| Full PPFL (FedAvg) [ppfl_clients_3] | nbaiot | 0.7523 | 0.4564 | 0.8039 | 0.9823 | 0.6803 | 0.0526 | 0.7309 | 0.9568 |
| Full PPFL (FedAvg) [ppfl_clients_5] | nbaiot | 0.7058 | 0.3656 | 0.7627 | 0.9829 | 0.6231 | 0.0465 | 0.6856 | 0.9501 |
| Full PPFL (FedAvg) [ppfl_local_epochs_1] | nbaiot | 0.4799 | 0.0064 | 0.5299 | 0.9686 | 0.3647 | 0.0506 | 0.4753 | 0.9136 |
| Full PPFL (FedAvg) [ppfl_local_epochs_2] | nbaiot | 0.6717 | 0.2662 | 0.7231 | 0.9794 | 0.5731 | 0.0516 | 0.6441 | 0.9371 |
| Full PPFL (FedAvg) [ppfl_local_epochs_5] | nbaiot | 0.7376 | 0.4558 | 0.7960 | 0.9824 | 0.6691 | 0.0514 | 0.7219 | 0.9568 |
| Full PPFL (FedAvg) [ppfl_sigma_0.4] | nbaiot | 0.7352 | 0.4538 | 0.7944 | 0.9824 | 0.6668 | 0.0512 | 0.7201 | 0.9561 |
| Full PPFL (FedAvg) [ppfl_sigma_0.6] | nbaiot | 0.7131 | 0.3751 | 0.7681 | 0.9815 | 0.6309 | 0.0510 | 0.6911 | 0.9483 |
| Full PPFL (FedAvg) [ppfl_sigma_0.8] | nbaiot | 0.6978 | 0.3359 | 0.7520 | 0.9807 | 0.6098 | 0.0515 | 0.6738 | 0.9421 |
| Full PPFL (FedAvg) [ppfl_sigma_1.0] | nbaiot | 0.6717 | 0.2662 | 0.7231 | 0.9794 | 0.5731 | 0.0516 | 0.6441 | 0.9371 |
| Full PPFL (FedAvg) [ppfl_sigma_1.5] | nbaiot | 0.5676 | 0.0495 | 0.6076 | 0.9736 | 0.4416 | 0.0514 | 0.5375 | 0.9229 |
| Full PPFL (FedAvg) [ppfl_sigma_2.5] | nbaiot | 0.4515 | 0.0032 | 0.5058 | 0.9670 | 0.3424 | 0.0500 | 0.4574 | 0.9109 |
| FL + SecAgg | nbaiot | 0.9976 | 0.9962 | 0.9926 | 0.9880 | 0.9973 | 0.0521 | 0.9879 | 0.9890 |
| Local-only | nbaiot | 0.9529 | 0.7316 | 0.9344 | 0.9809 | 0.8991 | 0.0758 | 0.9039 | 0.9674 |

## Privacy and cost

| Experiment | DP | SecAgg | ε (max client) | δ | σ | Total comm. (MiB) | Update (KiB) | SecAgg overhead (MiB) | Train time (s) | Client compute / round (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Centralized | no | no | – | – | – | 116.027 | – | – | 76.100 | – |
| FL + DP | yes | no | 7.481 | 0.000 | 1.000 | 32.016 | 78.903 | 0.000 | 456.668 | 2.142 |
| Plain FL (FedAvg) [fedavg] | no | no | – | – | – | 32.016 | 78.903 | 0.000 | 131.460 | 0.595 |
| Plain FL (FedAvg) [fedavg_alpha_0.1] | no | no | – | – | – | 32.016 | 78.903 | 0.000 | 128.060 | 0.579 |
| Plain FL (FedAvg) [fedavg_alpha_0.3] | no | no | – | – | – | 32.016 | 78.903 | 0.000 | 136.296 | 0.616 |
| Plain FL (FedAvg) [fedavg_alpha_1.0] | no | no | – | – | – | 32.016 | 78.903 | 0.000 | 134.384 | 0.607 |
| Plain FL (FedAvg) [fedavg_alpha_10.0] | no | no | – | – | – | 32.016 | 78.903 | 0.000 | 131.786 | 0.596 |
| FedProx | no | no | – | – | – | 32.016 | 78.903 | 0.000 | 170.706 | 0.790 |
| Full PPFL (FedAvg) [full_ppfl] | yes | yes | 7.481 | 0.000 | 1.000 | 50.150 | 157.753 | 2.304 | 433.282 | 1.936 |
| Full PPFL (FedProx) | yes | yes | 7.481 | 0.000 | 1.000 | 50.150 | 157.753 | 2.304 | 445.283 | 1.994 |
| Full PPFL (FedAvg) [full_ppfl_tuned] | yes | yes | 8.000 | 0.000 | 1.239 | 50.150 | 157.753 | 2.304 | 332.432 | 1.362 |
| Local-only | no | no | – | – | – | 0.000 | – | – | 65.272 | 5.803 |
| Full PPFL (FedAvg) [ppfl_clients_10] | yes | yes | 7.481 | 0.000 | 1.000 | 50.150 | 157.753 | 2.304 | 463.031 | 2.080 |
| Full PPFL (FedAvg) [ppfl_clients_20] | yes | yes | 13.087 | 0.000 | 1.000 | 104.996 | 157.753 | 9.082 | 517.540 | 0.956 |
| Full PPFL (FedAvg) [ppfl_clients_3] | yes | yes | 3.213 | 0.000 | 1.000 | 14.552 | 157.753 | 0.222 | 333.803 | 5.341 |
| Full PPFL (FedAvg) [ppfl_clients_5] | yes | yes | 4.282 | 0.000 | 1.000 | 24.488 | 157.753 | 0.593 | 419.651 | 3.980 |
| Full PPFL (FedAvg) [ppfl_local_epochs_1] | yes | yes | 5.254 | 0.000 | 1.000 | 50.149 | 157.753 | 2.304 | 284.397 | 1.170 |
| Full PPFL (FedAvg) [ppfl_local_epochs_2] | yes | yes | 7.481 | 0.000 | 1.000 | 50.149 | 157.753 | 2.304 | 444.287 | 1.998 |
| Full PPFL (FedAvg) [ppfl_local_epochs_5] | yes | yes | 12.443 | 0.000 | 1.000 | 50.149 | 157.753 | 2.304 | 948.252 | 4.517 |
| Full PPFL (FedAvg) [ppfl_sigma_0.4] | yes | yes | 88.634 | 0.000 | 0.400 | 50.149 | 157.753 | 2.304 | 447.827 | 1.891 |
| Full PPFL (FedAvg) [ppfl_sigma_0.6] | yes | yes | 26.089 | 0.000 | 0.600 | 50.150 | 157.753 | 2.305 | 494.956 | 2.146 |
| Full PPFL (FedAvg) [ppfl_sigma_0.8] | yes | yes | 12.189 | 0.000 | 0.800 | 50.149 | 157.753 | 2.304 | 445.753 | 1.961 |
| Full PPFL (FedAvg) [ppfl_sigma_1.0] | yes | yes | 7.481 | 0.000 | 1.000 | 50.149 | 157.753 | 2.304 | 406.683 | 1.812 |
| Full PPFL (FedAvg) [ppfl_sigma_1.5] | yes | yes | 3.773 | 0.000 | 1.500 | 50.149 | 157.753 | 2.304 | 509.031 | 2.334 |
| Full PPFL (FedAvg) [ppfl_sigma_2.5] | yes | yes | 1.921 | 0.000 | 2.500 | 50.150 | 157.753 | 2.304 | 393.101 | 1.782 |
| FL + SecAgg | no | yes | – | – | – | 50.150 | 157.753 | 2.305 | 153.994 | 0.606 |

Centralized communication is the size of the raw training/validation traffic that would have to be uploaded.

## Figures

- `results/figures/zero_day_recall_comparison.png`
- `results/figures/fpr_comparison.png`
- `results/figures/f1_comparison.png`
- `results/figures/mode_comparison.png`
- `results/figures/communication_comparison.png`
- `results/figures/f1_vs_epsilon.png`
- `results/figures/zero_day_recall_vs_privacy.png`
- `results/figures/f1_vs_communication.png`
- `results/figures/f1_convergence_overlay.png`
- `results/figures/val_loss_overlay.png`
- `results/figures/epsilon_overlay.png`
- `results/figures/sweep_scalability.png`
- `results/figures/sweep_heterogeneity.png`
- `results/figures/sweep_local_epochs.png`
