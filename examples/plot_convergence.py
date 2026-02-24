#!/usr/bin/env python3
"""
Simple plotting script for batch error analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('batch_error_analysis_results.csv')

# Calculate h = domain_length / n_elements (assuming domain_length = 1.0)
df['h'] = 1.0 / df['n_elements']

# Get unique dt values for separate curves
dt_values = sorted(df['dt'].unique())

# Create the plot
plt.figure(figsize=(10, 6))

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

for i, dt in enumerate(dt_values):
    # Filter data for this dt
    dt_data = df[df['dt'] == dt].copy()
    
    # Sort by h for smooth lines
    dt_data = dt_data.sort_values('h')
    
    # Plot trace error for equation 0 vs h
    plt.loglog(dt_data['h'], dt_data['trace_eq_0'], 
               'o-', color=colors[i % len(colors)], 
               label=f'dt = {dt}', markersize=4, linewidth=1.5)

plt.xlabel('h (spatial step size)')
plt.ylabel('Trace Error - Equation 0')
plt.title('Spatial Convergence: Trace Error vs h (fixed dt)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add reference lines for convergence rates
h_ref = np.array([1e-2, 1e-1])
plt.loglog(h_ref, h_ref, 'k--', alpha=0.5, label='O(h)')
plt.loglog(h_ref, h_ref**2, 'k:', alpha=0.5, label='O(h²)')
plt.legend()

plt.tight_layout()
plt.savefig('spatial_convergence_eq0.png', dpi=300, bbox_inches='tight')
plt.show()