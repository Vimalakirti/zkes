#!/usr/bin/env python3
"""
Simple script to collect criterion benchmark results and display them
No external dependencies required - uses only Python standard library
"""

import json
import os
import re
import math
from pathlib import Path

def read_criterion_results():
    """
    Read benchmark results from target/criterion/ directory
    """
    criterion_dir = Path("target/criterion")
    
    if not criterion_dir.exists():
        print(f"Error: {criterion_dir} directory not found.")
        print("Make sure you've run 'cargo bench --bench permute_prove' first.")
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

def analyze_results(results):
    """
    Analyze and display benchmark results
    """
    if not results:
        print("No benchmark results found.")
        return
    
    # Sort by size
    results_sorted = sorted(results)
    
    print("\n" + "="*70)
    print("PERMUTE PROVE FUNCTION BENCHMARK ANALYSIS")
    print("="*70)
    
    print(f"{'Size':<8} {'Time (ms)':<12} {'Elements':<10} {'Scaling':<10} {'Throughput (MB/s)':<15}")
    print("-" * 70)
    
    for i, (size, time_ms) in enumerate(results_sorted):
        elements = size * size
        
        # Calculate scaling factor
        if i > 0:
            prev_size, prev_time = results_sorted[i-1]
            time_ratio = time_ms / prev_time
            size_ratio = size / prev_size
            scaling = math.log(time_ratio) / math.log(size_ratio) if size_ratio > 1 else 0
            scaling_str = f"{scaling:.2f}"
        else:
            scaling_str = "-"
        
        # Calculate throughput (assuming 32 bytes per field element)
        bytes_per_element = 32
        total_bytes = elements * bytes_per_element
        time_seconds = time_ms / 1000.0
        throughput_mbs = (total_bytes / time_seconds) / (1024 * 1024)
        
        print(f"{size}×{size:<3} {time_ms:<12.3f} {elements:<10} {scaling_str:<10} {throughput_mbs:<15.2f}")
    
    # Overall statistics
    if len(results_sorted) > 1:
        scaling_factors = []
        for i in range(1, len(results_sorted)):
            prev_size, prev_time = results_sorted[i-1]
            size, time_ms = results_sorted[i]
            time_ratio = time_ms / prev_time
            size_ratio = size / prev_size
            scaling_factor = math.log(time_ratio) / math.log(size_ratio)
            scaling_factors.append(scaling_factor)
        
        avg_scaling = sum(scaling_factors) / len(scaling_factors)
        print(f"\nAverage scaling factor: {avg_scaling:.2f}")
        print("Interpretation:")
        print("  1.0 = Linear scaling (O(n))")
        print("  2.0 = Quadratic scaling (O(n²))")
        print("  3.0 = Cubic scaling (O(n³))")
    
    # Generate data for external plotting
    print(f"\n{'='*70}")
    print("CSV DATA FOR EXTERNAL PLOTTING")
    print("="*70)
    print("size,time_ms,elements")
    for size, time_ms in results_sorted:
        elements = size * size
        print(f"{size},{time_ms:.6f},{elements}")
    
    # Generate simple ASCII plot
    print(f"\n{'='*70}")
    print("ASCII PLOT (Time vs Elements)")
    print("="*70)
    
    if results_sorted:
        max_time = max(time for _, time in results_sorted)
        plot_width = 50
        
        for size, time_ms in results_sorted:
            elements = size * size
            bar_length = int((time_ms / max_time) * plot_width)
            bar = "█" * bar_length
            print(f"{elements:>7} |{bar:<{plot_width}} {time_ms:.3f}ms")
    
    print("\nTo create a proper plot:")
    print("1. Copy the CSV data above")
    print("2. Use Excel, Google Sheets, or any plotting tool")
    print("3. Plot 'elements' vs 'time_ms' as a line chart with linear scales")

def main():
    print("Criterion Benchmark Results Collector")
    print("=" * 40)
    
    # Read results from criterion directory
    results = read_criterion_results()
    
    if not results:
        print("\nNo benchmark results found.")
        print("To generate benchmark data:")
        print("1. Run: cargo bench --bench permute_prove")
        print("2. Then run this script again")
        return
    
    # Analyze and display results
    analyze_results(results)

if __name__ == "__main__":
    main()