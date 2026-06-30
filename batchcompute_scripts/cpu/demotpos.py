#!/usr/bin/env python3
"""
democpuH2-1.py
Demo workload for perf kernel tracing.
Generates:
 - high CPU load
 - cache pressure
 - frequent user -> kernel transitions
"""

import os
import time
import random

ARRAY_SIZE = 8_000_000       # Large array to stress cache
ITERATIONS = 30              # Total iterations
PAGE = 4096                  # Page size for memory access


def cpu_and_cache_work():
    # Allocate large array (cache pressure)
    data = bytearray(ARRAY_SIZE)

    total = 0
    # Strided access to cause cache misses
    for i in range(0, ARRAY_SIZE, PAGE):
        data[i] = (i * 7) & 0xFF
        total += data[i]

    return total


def syscall_work():
    # Repeated syscalls: getpid, stat, open/close
    for _ in range(200_000):
        os.getpid()
        try:
            fd = os.open("/dev/null", os.O_RDONLY)
            os.close(fd)
        except OSError:
            pass


def mixed_workload():
    for _ in range(ITERATIONS):
        cpu_and_cache_work()
        syscall_work()


def main():
    start = time.time()
    mixed_workload()
    end = time.time()

    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
