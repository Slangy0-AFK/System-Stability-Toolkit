#!/usr/bin/env python3
"""Workload Throttling Advisor: calculate conservative limits to reduce overload."""

import argparse
import json
import math
import os


K_MAX = 4096
B_MAX = 64
C_MAX = 2048
L_REF = 2048
A_0 = 500.0
A_0_GRAIN = 0.5
DEFAULT_MEMORY_PER_TASK_MB = 256.0


def detect_cpu_cores() -> int:
    """Return the number of CPU cores usable by this process."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def detect_available_memory_mb() -> float | None:
    """Return free memory in MB, or None when it cannot be detected."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (AttributeError, ValueError, OSError):
        return None


def get_buffer_window(current_scale: int) -> int:
    """Return the maximum amount of work kept in memory at once."""
    ratio = (1 + math.log(L_REF)) / (1 + math.log(current_scale))
    return max(1, min(K_MAX, current_scale, math.floor(K_MAX * ratio)))


def get_capacity_limit(
    current_scale: int,
    cpu_cores: int | None = None,
    available_memory_mb: float | None = None,
    memory_per_task_mb: float = DEFAULT_MEMORY_PER_TASK_MB,
) -> int:
    """Return the recommended maximum number of parallel tasks."""
    area_norm = math.sqrt(math.pi) / math.sqrt(math.log(current_scale) + A_0)
    limit = min(current_scale, math.floor(B_MAX * area_norm))
    if cpu_cores is not None:
        limit = min(limit, cpu_cores)
    if available_memory_mb is not None:
        limit = min(limit, math.floor(available_memory_mb / memory_per_task_mb))
    return max(1, limit)


def get_task_granularity(current_scale: int) -> int:
    """Return the recommended size for each small piece of work."""
    ln_scale = math.log(current_scale + 2)
    alpha = (1.0 / ln_scale) + (A_0_GRAIN / (ln_scale**2))
    chunk = math.ceil(C_MAX * alpha)
    return max(1, min(C_MAX, current_scale, chunk))


def get_recommendation(
    current_scale: int,
    cpu_cores: int | None = None,
    available_memory_mb: float | None = None,
    memory_per_task_mb: float = DEFAULT_MEMORY_PER_TASK_MB,
) -> dict[str, int]:
    """Return all three limits for a workload scale and machine resources."""
    if (
        isinstance(current_scale, bool)
        or not isinstance(current_scale, int)
        or current_scale < 1
    ):
        raise ValueError("workload scale must be a whole number of 1 or greater")
    if cpu_cores is not None and cpu_cores < 1:
        raise ValueError("CPU cores must be 1 or greater")
    if available_memory_mb is not None and available_memory_mb < 0:
        raise ValueError("available memory cannot be negative")
    if memory_per_task_mb <= 0:
        raise ValueError("memory per task must be greater than 0")

    return {
        "buffer_window": get_buffer_window(current_scale),
        "parallel_tasks": get_capacity_limit(
            current_scale, cpu_cores, available_memory_mb, memory_per_task_mb
        ),
        "task_size": get_task_granularity(current_scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate safer work limits for a computer under load."
    )
    parser.add_argument(
        "scale",
        type=int,
        help="estimated workload size, as a whole number of 1 or greater",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the result in a format other programs can read",
    )
    parser.add_argument(
        "--cpu-cores",
        type=int,
        help="CPU cores to plan for (default: detected)",
    )
    parser.add_argument(
        "--memory-mb",
        type=float,
        help="memory available for the job in MB (default: detected)",
    )
    parser.add_argument(
        "--memory-per-task-mb",
        type=float,
        default=DEFAULT_MEMORY_PER_TASK_MB,
        help="estimated memory used by each task in MB (default: %(default)s)",
    )
    args = parser.parse_args()
    cpu_cores = args.cpu_cores if args.cpu_cores is not None else detect_cpu_cores()
    memory_mb = (
        args.memory_mb if args.memory_mb is not None else detect_available_memory_mb()
    )
    try:
        recommendation = get_recommendation(
            args.scale, cpu_cores, memory_mb, args.memory_per_task_mb
        )
    except (ValueError, OverflowError) as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(recommendation, indent=2))
        return

    print(f"Recommended limits for workload scale {args.scale}:")
    print(f"  Keep at most {recommendation['buffer_window']} items waiting")
    print(f"  Run at most {recommendation['parallel_tasks']} tasks at once")
    print(f"  Process work in groups of {recommendation['task_size']}")


if __name__ == "__main__":
    main()