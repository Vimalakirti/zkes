#!/usr/bin/env python3
"""
Script to parse criterion benchmark output and plot performance vs matrix size

Requirements:
    pip install matplotlib numpy

Usage:
    # From benchmark output file:
    python plot_benchmark.py benchmark_output.txt
    
    # From clipboard/stdin:
    cargo bench --bench permute_prove | python plot_benchmark.py
    
    # Or pipe the output:
    python plot_benchmark.py < benchmark_output.txt
"""

import re
import sys

# Check for matplotlib and numpy
HAS_PLOTTING = True
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    HAS_PLOTTING = False
    print("Warning: Plotting packages not available.")
    print("To enable plotting, install with: pip install matplotlib numpy")
    print(f"Error: {e}")
    print("Will continue with text output only...")
    
    # Create dummy numpy for basic operations
    class DummyNumpy:
        @staticmethod
        def mean(x):
            return sum(x) / len(x)
        @staticmethod
        def log(x):
            import math
            return math.log(x)
    
    np = DummyNumpy()

def parse_benchmark_output(output_text):
    """
    Parse criterion benchmark output to extract size and timing data
    """
    sizes = []
    times = []
    
    # Pattern to match benchmark results
    # Example: "permute_prove/matrix_transpose/2x2"
    # followed by "time:   [5.1397 µs 5.1534 µs 5.1661 µs]"
    
    lines = output_text.split('\n')
    current_size = None
    
    for i, line in enumerate(lines):
        # Look for benchmark name with size
        size_match = re.search(r'matrix_transpose/(\d+)x\d+', line)
        if size_match:
            current_size = int(size_match.group(1))
        
        # Look for timing data on subsequent lines
        time_match = re.search(r'time:\s+\[[\d.]+\s*(\w+)\s+[\d.]+\s*\w+\s+[\d.]+\s*\w+\]', line)
        if time_match and current_size is not None:
            time_str = line
            # Extract the middle value (median)
            values = re.findall(r'([\d.]+)\s*(\w+)', time_str)
            if len(values) >= 2:  # We expect 3 values, take the middle one
                median_value = float(values[1][0])
                unit = values[1][1]
                
                # Convert to milliseconds for consistent units
                if unit == 'µs':
                    median_time_ms = median_value / 1000.0
                elif unit == 'ns':
                    median_time_ms = median_value / 1000000.0
                elif unit == 'ms':
                    median_time_ms = median_value
                elif unit == 's':
                    median_time_ms = median_value * 1000.0
                else:
                    median_time_ms = median_value  # Assume ms if unknown
                
                sizes.append(current_size)
                times.append(median_time_ms)
                current_size = None  # Reset for next benchmark
    
    return sizes, times

def print_text_analysis(sizes, times):
    """
    Print text-based analysis when plotting is not available
    """
    # Sort data by size
    sorted_data = sorted(zip(sizes, times))
    sizes_sorted, times_sorted = zip(*sorted_data)
    
    print("\n=== Benchmark Results ===")
    print("Size\t\tTime (ms)\tElements\tScaling")
    print("-" * 50)
    
    for i, (size, time) in enumerate(zip(sizes_sorted, times_sorted)):
        elements = size * size
        if i > 0:
            prev_size = sizes_sorted[i-1]
            prev_time = times_sorted[i-1]
            time_ratio = time / prev_time
            size_ratio = size / prev_size
            scaling = np.log(time_ratio) / np.log(size_ratio) if size_ratio > 1 else 0
            scaling_str = f"{scaling:.2f}"
        else:
            scaling_str = "-"
        
        print(f"{size}×{size}\t\t{time:.3f}\t\t{elements}\t\t{scaling_str}")
    
    # Calculate average scaling
    if len(sizes_sorted) > 1:
        scaling_factors = []
        for i in range(1, len(sizes_sorted)):
            size_ratio = sizes_sorted[i] / sizes_sorted[i-1]
            time_ratio = times_sorted[i] / times_sorted[i-1]
            scaling_factor = np.log(time_ratio) / np.log(size_ratio)
            scaling_factors.append(scaling_factor)
        
        avg_scaling = np.mean(scaling_factors)
        print(f"\nAverage scaling factor: {avg_scaling:.2f}")
        print("(Perfect quadratic scaling = 2.0, linear = 1.0)")
    
    # Create CSV output
    print(f"\n=== CSV Format ===")
    print("size,time_ms")
    for size, time in zip(sizes_sorted, times_sorted):
        print(f"{size},{time:.6f}")

def plot_benchmark_data(sizes, times):
    """
    Create a line plot of benchmark results
    """
    if not HAS_PLOTTING:
        print("Plotting not available. Showing text output only.")
        print_text_analysis(sizes, times)
        return
        
    plt.figure(figsize=(15, 10))
    
    # Sort data by size and convert to elements
    sorted_data = sorted(zip(sizes, times))
    sizes_sorted, times_sorted = zip(*sorted_data)
    elements_sorted = [s * s for s in sizes_sorted]
    time_per_element = [time / elements for time, elements in zip(times_sorted, elements_sorted)]
    
    # Create subplots to show different views
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Time vs Elements (linear scales)
    ax1.plot(elements_sorted, times_sorted, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Elements (N)', fontsize=10)
    ax1.set_ylabel('Time (ms)', fontsize=10)
    ax1.set_title('Time vs Elements (Linear Scale)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time per Element vs Elements (shows O(N) as horizontal line)
    ax2.plot(elements_sorted, time_per_element, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Elements (N)', fontsize=10)
    ax2.set_ylabel('Time per Element (ms/element)', fontsize=10)
    ax2.set_title('Time per Element vs Elements\n(Horizontal = O(N), Slope = Super-linear)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Log-log plot to see power law relationship
    ax3.loglog(elements_sorted, times_sorted, 'go-', linewidth=2, markersize=6, base=10)
    ax3.set_xlabel('Number of Elements (N) [log scale]', fontsize=10)
    ax3.set_ylabel('Time (ms) [log scale]', fontsize=10)
    ax3.set_title('Log-Log Plot\n(Slope = complexity exponent)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time vs Elements with O(N) reference line
    ax4.plot(elements_sorted, times_sorted, 'bo-', linewidth=2, markersize=6, label='Actual')
    # Add O(N) reference line using first data point
    ref_slope = times_sorted[0] / elements_sorted[0]
    ref_line = [ref_slope * elements for elements in elements_sorted]
    ax4.plot(elements_sorted, ref_line, 'r--', linewidth=2, alpha=0.7, label='Perfect O(N)')
    ax4.set_xlabel('Number of Elements (N)', fontsize=10)
    ax4.set_ylabel('Time (ms)', fontsize=10)
    ax4.set_title('Actual vs Perfect O(N) Scaling', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add data points annotations to key plots
    for i, (elements, time) in enumerate(zip(elements_sorted, times_sorted)):
        if i % 2 == 0 or elements > 100000:  # Annotate every other point and large points
            ax1.annotate(f'{elements}', (elements, time), textcoords="offset points", 
                        xytext=(0,5), ha='center', fontsize=8)
    
    # Calculate and show the actual complexity exponent
    import math
    if len(elements_sorted) > 1:
        # Use first and last points to estimate exponent
        log_ratio_time = math.log(times_sorted[-1] / times_sorted[0])
        log_ratio_elements = math.log(elements_sorted[-1] / elements_sorted[0])
        complexity_exponent = log_ratio_time / log_ratio_elements
        
        ax3.text(0.05, 0.95, f'Estimated exponent: {complexity_exponent:.3f}', 
                transform=ax3.transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('permute_benchmark_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis
    print("\n=== Complexity Analysis ===")
    print("Elements\t\tTime (ms)\tTime/Element (μs)\tExpected O(N)")
    print("-" * 70)
    
    baseline_ratio = times_sorted[0] / elements_sorted[0]  # Use first point as baseline
    
    for elements, time in zip(elements_sorted, times_sorted):
        time_per_element_us = (time / elements) * 1000  # Convert to microseconds
        expected_time = baseline_ratio * elements
        deviation_percent = ((time - expected_time) / expected_time) * 100
        
        print(f"{elements:>8}\t\t{time:>7.3f}\t\t{time_per_element_us:>8.3f}\t\t{deviation_percent:>+6.1f}%")
    
    if len(elements_sorted) > 1:
        print(f"\nComplexity Analysis:")
        print(f"Estimated exponent: {complexity_exponent:.3f}")
        if complexity_exponent < 1.1:
            print("✓ Algorithm is approximately O(N) - linear scaling!")
        elif complexity_exponent < 1.5:
            print("⚠ Algorithm is slightly super-linear but close to O(N)")
        else:
            print("✗ Algorithm is significantly super-linear")
    
    # Calculate scaling factor
    if len(sizes_sorted) > 1:
        scaling_factors = []
        for i in range(1, len(sizes_sorted)):
            size_ratio = sizes_sorted[i] / sizes_sorted[i-1]
            time_ratio = times_sorted[i] / times_sorted[i-1]
            scaling_factor = np.log(time_ratio) / np.log(size_ratio)
            scaling_factors.append(scaling_factor)
        
        avg_scaling = np.mean(scaling_factors)
        print(f"\nAverage scaling factor: {avg_scaling:.2f}")
        print(f"(Perfect quadratic scaling would be 2.0)")

def read_criterion_results():
    """
    Read benchmark results from target/criterion/ directory
    """
    import os
    import json
    from pathlib import Path
    
    criterion_dir = Path("target/criterion")
    
    if not criterion_dir.exists():
        print(f"Error: {criterion_dir} directory not found.")
        print("Make sure you've run 'cargo bench' first.")
        return []
    
    results = []
    
    # Look for both permute_prove and permute_prove_large benchmark results
    benchmark_configs = [
        ("permute_prove", "matrix_transpose"),
        ("permute_prove_large", "matrix_transpose_large")
    ]
    
    for bench_name, matrix_name in benchmark_configs:
        permute_dir = criterion_dir / bench_name / matrix_name
        
        if permute_dir.exists():
            print(f"Processing {bench_name}/{matrix_name}...")
            
            for size_dir in permute_dir.iterdir():
                if size_dir.is_dir() and size_dir.name != "report":
                    # Look for estimates.json file
                    estimates_file = size_dir / "base" / "estimates.json"
                    if estimates_file.exists():
                        try:
                            with open(estimates_file, 'r') as f:
                                data = json.load(f)
                            
                            # Extract size from directory name (e.g., "2x2", "4x4", "512x512", etc.)
                            dir_name = size_dir.name
                            size_match = re.search(r'(\d+)x\d+', dir_name)
                            if size_match:
                                size = int(size_match.group(1))
                                
                                # Get median time in nanoseconds and convert to milliseconds
                                median_ns = data.get("median", {}).get("point_estimate", 0)
                                median_ms = median_ns / 1_000_000.0  # Convert ns to ms
                                
                                results.append((size, median_ms))
                                print(f"  Found: {dir_name} -> {median_ms:.3f}ms")
                                
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"  Error reading {estimates_file}: {e}")
                    else:
                        print(f"  No estimates.json found in {size_dir}")
        else:
            print(f"Directory {permute_dir} not found, skipping...")
    
    return results

def main():
    print("Permute Benchmark Data Collector and Plotter")
    print("=" * 50)
    
    # First try to read from criterion directory
    results = read_criterion_results()
    
    if results:
        sizes, times = zip(*results)
        print(f"Found {len(results)} benchmark results from target/criterion/")
    elif len(sys.argv) > 1:
        # Fallback: Read from file if provided
        filename = sys.argv[1]
        try:
            with open(filename, 'r') as f:
                output_text = f.read()
            sizes, times = parse_benchmark_output(output_text)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            return
    else:
        print("No criterion results found and no input file provided.")
        print("Usage options:")
        print("1. Run 'cargo bench --bench permute_prove' first, then run this script")
        print("2. Provide a benchmark output file: python plot_benchmark.py output.txt")
        return
    
    if not sizes:
        print("No benchmark data found.")
        return
    
    print(f"Processing {len(sizes)} benchmark results")
    
    # Create the plot
    plot_benchmark_data(sizes, times)
    
    if HAS_PLOTTING:
        print("Plot saved as 'permute_benchmark_plot.png'")
    else:
        print("Plotting not available - showed text analysis only")

if __name__ == "__main__":
    main()