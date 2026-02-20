#!/usr/bin/env python3
"""
Batch Error Analysis Script for BioNetFlux

This script runs convergence analysis by varying dt and n_elements parameters
in the KS problem configuration and collecting error results.
"""

import sys
import os
import subprocess
import itertools
import toml
import csv
import time
from datetime import datetime
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def modify_toml_parameters(config_file: str, dt: float, n_elements: int):
    """
    Modify the TOML configuration file with new dt and n_elements values.
    
    Args:
        config_file: Path to the TOML configuration file
        dt: New time step value
        n_elements: New number of elements value
    """
    # Read current configuration
    with open(config_file, 'r') as f:
        config = toml.load(f)
    
    # Update parameters
    config['time_parameters']['dt'] = dt
    config['discretization']['n_elements'] = n_elements
    
    # Write back to file
    with open(config_file, 'w') as f:
        toml.dump(config, f)
    
    print(f"Updated config: dt={dt}, n_elements={n_elements}")


def run_evolution_example(config_file: str):
    """
    Run the evolution example with the specified configuration file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        tuple: (return_code, stdout, stderr)
    """
    script_path = os.path.join(os.path.dirname(__file__), "evolution_example_ks.py")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path, config_file],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.path.dirname(__file__)
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Process timed out after 5 minutes"
    except Exception as e:
        return -1, "", str(e)


def parse_error_results(stdout: str, dt: float, n_elements: int):
    """
    Parse error results from the evolution example output.
    
    Args:
        stdout: Standard output from the evolution example
        dt: Time step used
        n_elements: Number of elements used
        
    Returns:
        dict: Parsed error information
    """
    lines = stdout.split('\n')
    
    result = {
        'dt': dt,
        'n_elements': n_elements,
        'final_time': None,
        'successful_steps': None,
        'trace_errors': {},
        'bulk_errors': {},
        'has_errors': False
    }
    
    # Extract basic run information
    for line in lines:
        if "Evolution completed:" in line:
            # Extract final time and steps
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "time:" and i + 1 < len(parts):
                    result['final_time'] = float(parts[i + 1])
                if "/" in part and "steps" in parts[i + 1:i + 3]:
                    steps_info = part.split('/')
                    if len(steps_info) == 2:
                        result['successful_steps'] = int(steps_info[0])
    
    # Parse error sections
    in_trace_section = False
    in_bulk_section = False
    
    for line in lines:
        # Check for error section headers
        if "TRACE ERRORS (weighted Euclidean):" in line:
            in_trace_section = True
            in_bulk_section = False
            continue
        elif "BULK ERRORS (L2 norm):" in line:
            in_trace_section = False
            in_bulk_section = True
            continue
        elif line.strip() == "":
            # Empty line might end a section
            continue
        elif not line.startswith("  ") and line.strip():
            # Non-indented line that's not empty ends the current section
            if "Error analysis completed successfully" in line:
                break
            in_trace_section = False
            in_bulk_section = False
            continue
        
        # Parse error values
        if in_trace_section or in_bulk_section:
            # Look for lines starting with "  Equation"
            if line.strip().startswith("Equation"):
                try:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        eq_num = int(parts[1].rstrip(':'))
                        error_val = float(parts[2])
                        
                        if in_trace_section:
                            result['trace_errors'][f'eq_{eq_num}'] = error_val
                        else:  # in_bulk_section
                            result['bulk_errors'][f'eq_{eq_num}'] = error_val
                        
                        result['has_errors'] = True
                        
                except (ValueError, IndexError):
                    continue
    
    return result


def save_results_to_csv(results: list, output_file: str):
    """
    Save error analysis results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_file: Output CSV file path
    """
    if not results:
        print("No results to save")
        return
    
    # Determine all possible equation indices
    all_trace_eqs = set()
    all_bulk_eqs = set()
    
    for result in results:
        all_trace_eqs.update(result['trace_errors'].keys())
        all_bulk_eqs.update(result['bulk_errors'].keys())
    
    # Sort equation keys for consistent ordering
    all_trace_eqs = sorted(all_trace_eqs)
    all_bulk_eqs = sorted(all_bulk_eqs)
    
    # Prepare CSV headers
    headers = ['dt', 'n_elements', 'final_time', 'successful_steps', 'run_status']
    headers.extend([f'trace_{eq}' for eq in all_trace_eqs])
    headers.extend([f'bulk_{eq}' for eq in all_bulk_eqs])
    
    # Write CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for result in results:
            row = {
                'dt': result['dt'],
                'n_elements': result['n_elements'],
                'final_time': result.get('final_time', 'N/A'),
                'successful_steps': result.get('successful_steps', 'N/A'),
                'run_status': 'success' if result['has_errors'] else 'no_errors'
            }
            
            # Add trace errors
            for eq in all_trace_eqs:
                row[f'trace_{eq}'] = result['trace_errors'].get(eq, 'N/A')
            
            # Add bulk errors
            for eq in all_bulk_eqs:
                row[f'bulk_{eq}'] = result['bulk_errors'].get(eq, 'N/A')
            
            writer.writerow(row)
    
    print(f"Results saved to {output_file}")


def main():
    """Main batch analysis function."""
    
    print("=== BioNetFlux Batch Error Analysis ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration parameters
    dt_values = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125]
    n_elements_values = [10, 20, 40, 80, 160]
    
    # File paths
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    config_file = os.path.join(config_dir, "ks_parameters.toml")
    backup_config = os.path.join(config_dir, "ks_parameters_backup.toml")
    output_file = os.path.join(os.path.dirname(__file__), "batch_error_analysis_results.csv")
    
    # Create backup of original config
    if not os.path.exists(backup_config):
        import shutil
        shutil.copy2(config_file, backup_config)
        print(f"Created backup: {backup_config}")
    
    # Verify files exist
    if not os.path.exists(config_file):
        print(f"ERROR: Configuration file not found: {config_file}")
        return
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(dt_values, n_elements_values))
    total_runs = len(param_combinations)
    
    print(f"Running {total_runs} parameter combinations:")
    print(f"  dt values: {dt_values}")
    print(f"  n_elements values: {n_elements_values}")
    print()
    
    # Storage for results
    all_results = []
    successful_runs = 0
    failed_runs = 0
    
    # Run analysis for each combination
    for i, (dt, n_elements) in enumerate(param_combinations, 1):
        print(f"Run {i}/{total_runs}: dt={dt}, n_elements={n_elements}")
        print("-" * 50)
        
        try:
            # Update configuration file
            modify_toml_parameters(config_file, dt, n_elements)
            
            # Run evolution example
            return_code, stdout, stderr = run_evolution_example(config_file)
            
            if return_code == 0:
                # Parse results
                result = parse_error_results(stdout, dt, n_elements)
                all_results.append(result)
                
                if result['has_errors']:
                    print(f"✓ Success: Computed errors for dt={dt}, n_elements={n_elements}")
                    successful_runs += 1
                else:
                    print(f"⚠ Warning: No errors computed (likely missing analytical solutions)")
                    successful_runs += 1
            else:
                print(f"✗ Failed: Return code {return_code}")
                if stderr:
                    print(f"Error output: {stderr[:200]}...")
                failed_runs += 1
                
                # Still add a result entry for tracking
                result = {
                    'dt': dt,
                    'n_elements': n_elements,
                    'final_time': None,
                    'successful_steps': None,
                    'trace_errors': {},
                    'bulk_errors': {},
                    'has_errors': False
                }
                all_results.append(result)
        
        except Exception as e:
            print(f"✗ Exception occurred: {e}")
            failed_runs += 1
        
        print()
        time.sleep(1)  # Brief pause between runs
    
    # Save all results
    save_results_to_csv(all_results, output_file)
    
    # Restore original configuration
    if os.path.exists(backup_config):
        import shutil
        shutil.copy2(backup_config, config_file)
        print(f"Restored original configuration from backup")
    
    # Summary
    print("=== BATCH ANALYSIS SUMMARY ===")
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Results saved to: {output_file}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show error summary if available
    error_results = [r for r in all_results if r['has_errors']]
    if error_results:
        print(f"\nError computation successful for {len(error_results)} parameter combinations")
        print("Error range summary:")
        
        # Calculate error ranges
        all_trace_errors = []
        all_bulk_errors = []
        
        for result in error_results:
            all_trace_errors.extend(result['trace_errors'].values())
            all_bulk_errors.extend(result['bulk_errors'].values())
        
        if all_trace_errors:
            print(f"  Trace errors: [{min(all_trace_errors):.2e}, {max(all_trace_errors):.2e}]")
        if all_bulk_errors:
            print(f"  Bulk errors:  [{min(all_bulk_errors):.2e}, {max(all_bulk_errors):.2e}]")


if __name__ == "__main__":
    main()
