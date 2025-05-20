# Cluster Algorithms for the 2D Ising Model

This project compares the performance of three Monte Carlo algorithms—Metropolis, Wolff, and Swendsen–Wang—in simulating the 2D Ising model. The focus is on evaluating runtime efficiency, autocorrelation times, and the effects of critical slowing down.

## Overview

- **Model**: 2D Ising model on a square lattice with periodic boundary conditions
- **Algorithms**: Metropolis (with/without sub-sweeps), Wolff, Swendsen–Wang
- **Observables**: Magnetization, Magnetic Susceptibility, Binder Cumulant
- **Analysis**: Binning analysis, runtime benchmarking, and autocorrelation time estimation
- **Critical temperature**: Close to $T_c \approx 2.269$

## Key Findings

- The **Wolff algorithm** achieves the best performance near criticality, with minimal autocorrelation time and high MCspeed.
- **Swendsen–Wang** also reduces autocorrelations effectively, but has higher runtime due to multiple cluster handling.
- **Metropolis**, even with sub-sweeps, shows significantly slower convergence and poor sampling efficiency near $T_c$.

## Implementation Notes

- Linear autocorrelation times are estimated using binning analysis of magnetization data.
- Results are compared across lattice sizes $L = 8, 10, 12, 14, 16$ and temperatures near $T_c$.
- Sub-sweeps are applied to Metropolis to ensure a fair comparison with cluster updates.
- Cluster algorithms are implemented without sub-sweeps due to their intrinsic efficiency.

## Conclusion

Cluster algorithms, particularly Wolff, demonstrate significant advantages for studying phase transitions in the 2D Ising model. They outperform the Metropolis method in both speed and accuracy, especially in the critical region.
