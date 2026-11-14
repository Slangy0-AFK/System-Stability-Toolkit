import unittest

from crash_fix import (
    get_buffer_window,
    get_capacity_limit,
    get_recommendation,
    get_task_granularity,
)


class CrashFixTests(unittest.TestCase):
    def test_small_workload_is_capped_to_workload(self):
        self.assertEqual(
            get_recommendation(1),
            {"buffer_window": 1, "parallel_tasks": 1, "task_size": 1},
        )

    def test_no_jump_between_neighbouring_workloads(self):
        for workload in range(1, 5000):
            low, high = get_recommendation(workload), get_recommendation(workload + 1)
            for key in low:
                self.assertLessEqual(abs(high[key] - low[key]), max(2, low[key] * 0.1))

    def test_limits_never_exceed_workload_or_built_in_caps(self):
        for workload in list(range(1, 300)) + [2048, 10**6, 10**1000]:
            r = get_recommendation(workload, cpu_cores=8, available_memory_mb=1e6)
            self.assertLessEqual(r["buffer_window"], min(workload, 4096))
            self.assertLessEqual(r["parallel_tasks"], min(workload, 64, 8))
            self.assertLessEqual(r["task_size"], min(workload, 2048))
            self.assertTrue(all(v >= 1 for v in r.values()))

    def test_cpu_and_memory_limit_parallelism(self):
        self.assertEqual(get_capacity_limit(2048, cpu_cores=2), 2)
        self.assertEqual(get_capacity_limit(2048, available_memory_mb=512), 2)
        self.assertEqual(
            get_capacity_limit(2048, available_memory_mb=512, memory_per_task_mb=100),
            5,
        )
        self.assertEqual(get_capacity_limit(2048, available_memory_mb=10), 1)

    def test_invalid_resources_are_rejected(self):
        with self.assertRaises(ValueError):
            get_recommendation(10, cpu_cores=0)
        with self.assertRaises(ValueError):
            get_recommendation(10, memory_per_task_mb=0)
        with self.assertRaises(ValueError):
            get_recommendation(10, available_memory_mb=-1)

    def test_large_workload_returns_valid_limits(self):
        recommendation = get_recommendation(1_000_000)
        self.assertGreaterEqual(recommendation["buffer_window"], 1)
        self.assertGreaterEqual(recommendation["parallel_tasks"], 1)
        self.assertGreaterEqual(recommendation["task_size"], 1)

    def test_invalid_workload_is_rejected(self):
        with self.assertRaises(ValueError):
            get_recommendation(0)
        with self.assertRaises(ValueError):
            get_recommendation(1.5)
        with self.assertRaises(ValueError):
            get_recommendation(True)

    def test_recommendations_stay_within_built_in_limits(self):
        for workload in (1, 2, 2048, 1_000_000, 10**1000):
            recommendation = get_recommendation(workload)
            self.assertLessEqual(recommendation["buffer_window"], 4096)
            self.assertLessEqual(recommendation["parallel_tasks"], 64)
            self.assertLessEqual(recommendation["task_size"], 2048)
            self.assertTrue(all(value >= 1 for value in recommendation.values()))


if __name__ == "__main__":
    unittest.main()