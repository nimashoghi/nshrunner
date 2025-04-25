from __future__ import annotations

import types

from nshrunner.wrapper_fns import submit_parallel_screen


def _test_fn(x: int) -> int:
    """Test function that doubles its input."""
    return x * 2


def test_submit_parallel_screen_validation():
    # Create test arguments with explicit type hint
    args_list: list[tuple[int]] = [(i,) for i in range(5)]

    # Test with invalid num_parallel_screens
    try:
        submit_parallel_screen(_test_fn, args_list, num_parallel_screens=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "num_parallel_screens must be positive"

    # Test with num_parallel_screens > len(args_list)
    submissions = submit_parallel_screen(_test_fn, args_list, num_parallel_screens=10)
    assert len(submissions) == len(args_list)  # Should be capped at args_list length


def test_submit_parallel_screen_partitioning():
    """Test round-robin scheduling logic of submit_parallel_screen."""
    from nshrunner import wrapper_fns

    class DummyRunner:
        def __init__(self, fn, config):
            self.fn = fn
            self.config = config

        def session(self, partition, kwargs):
            # Return an object with partition, kwargs, and snapshot attributes
            return types.SimpleNamespace(
                partition=partition, kwargs=kwargs, snapshot=None
            )

    original_runner = wrapper_fns.Runner
    wrapper_fns.Runner = DummyRunner
    try:
        args_list: list[tuple[int]] = [(i,) for i in range(6)]
        # For 2 parallel screens with round_robin scheduling (default)
        submissions = wrapper_fns.submit_parallel_screen(
            _test_fn, args_list, num_parallel_screens=2, scheduling="round_robin"
        )
        assert len(submissions) == 2
        expected_partitions = [
            args_list[0::2],  # [ (0,), (2,), (4,) ]
            args_list[1::2],  # [ (1,), (3,), (5,) ]
        ]
        for sub, expected in zip(submissions, expected_partitions):
            # No longer need cast(Any, sub) as it's a SimpleNamespace
            assert getattr(sub, "partition") == expected, (
                f"Expected {expected}, got {getattr(sub, 'partition')}"
            )
    finally:
        wrapper_fns.Runner = original_runner


def test_submit_parallel_screen_block_scheduling():
    """Test block scheduling logic of submit_parallel_screen."""
    from nshrunner import wrapper_fns

    class DummyRunner:
        def __init__(self, fn, config):
            self.fn = fn
            self.config = config

        def session(self, partition, kwargs):
            # Return an object with partition, kwargs, and snapshot attributes
            return types.SimpleNamespace(
                partition=partition, kwargs=kwargs, snapshot=None
            )

    original_runner = wrapper_fns.Runner
    wrapper_fns.Runner = DummyRunner
    try:
        args_list: list[tuple[int]] = [(i,) for i in range(6)]
        # For 2 parallel screens with block scheduling: chunks of size 3.
        submissions = wrapper_fns.submit_parallel_screen(
            _test_fn, args_list, num_parallel_screens=2, scheduling="block"
        )
        assert len(submissions) == 2
        expected_partitions = [
            args_list[0:3],  # [(0,), (1,), (2,)]
            args_list[3:6],  # [(3,), (4,), (5,)]
        ]
        for sub, expected in zip(submissions, expected_partitions):
            # No longer need cast(Any, sub) as it's a SimpleNamespace
            assert getattr(sub, "partition") == expected, (
                f"Expected {expected}, got {getattr(sub, 'partition')}"
            )
    finally:
        wrapper_fns.Runner = original_runner
