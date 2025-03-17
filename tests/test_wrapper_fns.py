from __future__ import annotations

from typing import Any, cast

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
    """Test partitioning logic of submit_parallel_screen."""
    from nshrunner import wrapper_fns

    # Dummy Runner to capture partitioned arguments
    class DummyRunner:
        def __init__(self, fn, config):
            self.fn = fn
            self.config = config

        def session(self, partition, kwargs):
            return {"partition": partition, "kwargs": kwargs}

    # Save the original Runner and patch it
    original_runner = wrapper_fns.Runner
    wrapper_fns.Runner = DummyRunner

    try:
        args_list: list[tuple[int]] = [(i,) for i in range(6)]
        # For 2 parallel screens, partitions should be: [ (0,), (2,), (4,) ] and [ (1,), (3,), (5,) ]
        submissions = wrapper_fns.submit_parallel_screen(
            _test_fn, args_list, num_parallel_screens=2
        )
        assert len(submissions) == 2
        expected_partitions = [
            args_list[0::2],  # indices 0,2,4
            args_list[1::2],  # indices 1,3,5
        ]
        for sub, expected in zip(submissions, expected_partitions):
            sub = cast(Any, sub)
            assert (
                sub["partition"] == expected
            ), f"Expected {expected}, got {sub['partition']}"
    finally:
        # Restore the original Runner
        wrapper_fns.Runner = original_runner
