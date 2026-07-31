import pytest

from utils.runner_setup.plan_node_labels import plan_node_labels


def test_twelve_runner_pool_has_floor_capacity_slots_per_node_count():
    runners = [f"gpu_{index:02d}" for index in range(12)]

    plan = plan_node_labels(runners)

    for node_count in range(1, 13):
        assert sum(
            f"nodes:{node_count}" in labels for labels in plan.values()
        ) == 12 // node_count


def test_larger_allocations_are_nested_on_earlier_runners():
    plan = plan_node_labels(["gpu_00", "gpu_01", "gpu_02", "gpu_03"])

    assert plan == {
        "gpu_00": ["nodes:1", "nodes:2", "nodes:3", "nodes:4"],
        "gpu_01": ["nodes:1", "nodes:2"],
        "gpu_02": ["nodes:1"],
        "gpu_03": ["nodes:1"],
    }


def test_pool_must_be_nonempty_and_unique():
    with pytest.raises(ValueError, match="empty"):
        plan_node_labels([])
    with pytest.raises(ValueError, match="duplicate"):
        plan_node_labels(["gpu_00", "gpu_00"])
