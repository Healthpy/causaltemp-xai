"""Example demonstrating the new temporal sparsity metrics."""

import numpy as np
from causaltemp_xai.metrics.axis_c import sparsity

# Example: temporal data with shape (T, k)
x_orig = np.zeros((4, 3))  # 4 timepoints, 3 channels
x_cf = x_orig.copy()

# Modify entire first channel
x_cf[:, 0] = 1.0

# Default behavior (backward compatible): returns scalar
overall_sparsity = sparsity(x_orig, x_cf)
print(f"Overall sparsity (scalar): {overall_sparsity:.2f}")

# New detailed metrics
detailed = sparsity(x_orig, x_cf, return_detailed=True)
print(f"\nDetailed sparsity breakdown:")
print(f"  Channel sparsity: {detailed['channels']:.2f}")
print(f"  Timepoint sparsity: {detailed['timepoints']:.2f}")

print("\nInterpretation:")
print(f"  - {detailed['channels']*100:.0f}% of channels are entirely unchanged (channels 1,2 untouched)")
print(f"  - {detailed['timepoints']*100:.0f}% of timepoints are entirely unchanged (all have channel 0 modified)")

# More complex example
print("\n" + "="*60)
print("Complex example: mixed modifications")
print("="*60)

x_orig2 = np.zeros((5, 4))
x_cf2 = x_orig2.copy()

# Modify only channel 0 entirely (all timepoints, only channel 0)
x_cf2[:, 0] = 1.0

detailed2 = sparsity(x_orig2, x_cf2, return_detailed=True)
print(f"\nModified only channel 0 (all timepoints):")
print(f"  Channel sparsity: {detailed2['channels']:.2f} (only channels 1-3 untouched → 3/4)")
print(f"  Timepoint sparsity: {detailed2['timepoints']:.2f} (no timepoint entirely unchanged)")
