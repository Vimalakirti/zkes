#!/usr/bin/env python3
"""
Parse timing logs and accumulate constant commit times.

Usage: python parse_timing.py output.log
"""

import re
import sys
from datetime import timedelta

def parse_timing_log(filename):
    """Parse the log file and accumulate constant commit times."""
    
    # Regex pattern to match constant commit timing lines
    pattern0 = r'\[.*\] \| ([\d.]+)s to commit'
    pattern1 = r'\[.*\] \| ([\d.]+)s to prove'
    pattern = r'\[.*\] \| \| ([\d.]+)s to commit to constant for edge (\d+)'
    
    total_time = 0.0
    total_time0 = 0.0
    total_time1 = 0.0
    constant_commits = []
    
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                match = re.search(pattern, line.strip())
                if match:
                    time_seconds = float(match.group(1))
                    edge_id = int(match.group(2))
                    
                    total_time += time_seconds
                    constant_commits.append((edge_id, time_seconds))
                    
                    print(f"Edge {edge_id}: {time_seconds}s")
                match0 = re.search(pattern0, line.strip())
                if match0:
                    time_seconds = float(match0.group(1))
                    total_time0 += time_seconds

                match1 = re.search(pattern1, line.strip())
                if match1:
                    time_seconds = float(match1.group(1))
                    total_time1 += time_seconds
        
        others = total_time0 - total_time
        prover = total_time1 + others
        print(f"\n--- Summary ---")
        print(f"Total constant commits found: {len(constant_commits)}")
        print(f"Total time for constant commits: {total_time:.6f}s")
        print(f"Total time for others commits: {others:.6f}s")
        print(f"Total time for prover: {prover:.6f}s")
        
        if constant_commits:
            avg_time = total_time / len(constant_commits)
            print(f"Average time per constant commit: {avg_time:.6f}s")
            
            # Find slowest and fastest commits
            slowest = max(constant_commits, key=lambda x: x[1])
            fastest = min(constant_commits, key=lambda x: x[1])
            print(f"Slowest commit: Edge {slowest[0]} ({slowest[1]}s)")
            print(f"Fastest commit: Edge {fastest[0]} ({fastest[1]}s)")
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except Exception as e:
        print(f"Error parsing file: {e}")
        return

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_timing.py <log_file>")
        print("Example: python parse_timing.py output.log")
        sys.exit(1)
    
    log_file = sys.argv[1]
    parse_timing_log(log_file)

if __name__ == "__main__":
    main()