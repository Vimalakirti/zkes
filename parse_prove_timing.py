#!/usr/bin/env python3
"""
Parse timing logs and collect prove node timing information.

Extracts timing information in the format:
"0.0961s to prove node 1428 | kind Einsum(Einsum { equation: "bsi,ij->bsj" })"

Usage: python parse_prove_timing.py logfile.log
"""

import re
import sys
from collections import defaultdict, Counter

def parse_prove_timing_log(filename):
    """Parse the log file and collect prove node timing information."""
    
    # Regex pattern to match prove node timing lines
    # Format: {time}s to prove node {node_id} | kind {operation_type}
    pattern = r'([\d.]+)s to prove node (\d+) \| kind (.+)'
    
    prove_timings = []
    operation_stats = defaultdict(list)
    node_timings = {}
    
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                match = re.search(pattern, line.strip())
                if match:
                    time_seconds = float(match.group(1))
                    node_id = int(match.group(2))
                    operation_kind = match.group(3).strip()
                    
                    # Store the timing information
                    timing_entry = {
                        'time': time_seconds,
                        'node_id': node_id,
                        'operation': operation_kind,
                        'line_num': line_num
                    }
                    
                    prove_timings.append(timing_entry)
                    operation_stats[operation_kind].append(time_seconds)
                    node_timings[node_id] = time_seconds
                    
                    print(f"Node {node_id}: {time_seconds}s | {operation_kind}")
        
        # Print detailed statistics
        print_statistics(prove_timings, operation_stats, node_timings)
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except Exception as e:
        print(f"Error parsing file: {e}")
        return

def print_statistics(prove_timings, operation_stats, node_timings):
    """Print detailed statistics about the prove timings."""
    
    if not prove_timings:
        print("No prove timing entries found.")
        return
    
    print(f"\n--- Summary ---")
    print(f"Total prove operations found: {len(prove_timings)}")
    
    # Total time
    total_time = sum(entry['time'] for entry in prove_timings)
    print(f"Total prove time: {total_time:.6f}s")
    
    # Average time
    avg_time = total_time / len(prove_timings)
    print(f"Average time per prove operation: {avg_time:.6f}s")
    
    # Find slowest and fastest operations
    slowest = max(prove_timings, key=lambda x: x['time'])
    fastest = min(prove_timings, key=lambda x: x['time'])
    print(f"\nSlowest operation: Node {slowest['node_id']} ({slowest['time']}s)")
    print(f"  Operation: {slowest['operation']}")
    print(f"Fastest operation: Node {fastest['node_id']} ({fastest['time']}s)")
    print(f"  Operation: {fastest['operation']}")
    
    # Statistics by operation type
    print(f"\n--- Operation Type Statistics ---")
    for operation, times in operation_stats.items():
        count = len(times)
        total_op_time = sum(times)
        avg_op_time = total_op_time / count
        max_op_time = max(times)
        min_op_time = min(times)
        
        print(f"\n{operation}:")
        print(f"  Count: {count}")
        print(f"  Total time: {total_op_time:.6f}s ({total_op_time/total_time*100:.1f}% of total)")
        print(f"  Average time: {avg_op_time:.6f}s")
        print(f"  Min time: {min_op_time:.6f}s")
        print(f"  Max time: {max_op_time:.6f}s")
    
    # Top 10 slowest nodes
    print(f"\n--- Top 10 Slowest Nodes ---")
    sorted_timings = sorted(prove_timings, key=lambda x: x['time'], reverse=True)
    for i, entry in enumerate(sorted_timings[:10]):
        print(f"{i+1:2d}. Node {entry['node_id']}: {entry['time']:.6f}s | {entry['operation']}")
    
    # Extract and show unique equation types for Einsum operations
    einsum_equations = set()
    for entry in prove_timings:
        operation = entry['operation']
        # Extract equation from Einsum operations
        einsum_match = re.search(r'Einsum.*equation:\s*"([^"]+)"', operation)
        if einsum_match:
            equation = einsum_match.group(1)
            einsum_equations.add(equation)
    
    if einsum_equations:
        print(f"\n--- Unique Einsum Equations Found ---")
        for i, equation in enumerate(sorted(einsum_equations), 1):
            print(f"{i:2d}. {equation}")
    
    # Show distribution of operation types
    operation_counts = Counter(entry['operation'] for entry in prove_timings)
    print(f"\n--- Operation Type Distribution ---")
    for operation, count in operation_counts.most_common():
        percentage = count / len(prove_timings) * 100
        print(f"{count:4d} ({percentage:5.1f}%) | {operation}")

def export_csv(prove_timings, output_file):
    """Export timing data to CSV format."""
    import csv
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['node_id', 'time_seconds', 'operation', 'line_num']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in prove_timings:
            writer.writerow({
                'node_id': entry['node_id'],
                'time_seconds': entry['time'],
                'operation': entry['operation'],
                'line_num': entry['line_num']
            })
    
    print(f"Timing data exported to: {output_file}")

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python parse_prove_timing.py <log_file> [--csv]")
        print("Example: python parse_prove_timing.py output.log")
        print("         python parse_prove_timing.py output.log --csv")
        sys.exit(1)
    
    log_file = sys.argv[1]
    export_csv_flag = len(sys.argv) == 3 and sys.argv[2] == '--csv'
    
    # Parse the timing log
    parse_prove_timing_log(log_file)
    
    # If CSV export is requested, parse again and export
    if export_csv_flag:
        pattern = r'([\d.]+)s to prove node (\d+) \| kind (.+)'
        prove_timings = []
        
        try:
            with open(log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    match = re.search(pattern, line.strip())
                    if match:
                        time_seconds = float(match.group(1))
                        node_id = int(match.group(2))
                        operation_kind = match.group(3).strip()
                        
                        timing_entry = {
                            'time': time_seconds,
                            'node_id': node_id,
                            'operation': operation_kind,
                            'line_num': line_num
                        }
                        prove_timings.append(timing_entry)
            
            csv_file = log_file.replace('.log', '_timing.csv')
            export_csv(prove_timings, csv_file)
            
        except Exception as e:
            print(f"Error exporting CSV: {e}")

if __name__ == "__main__":
    main()