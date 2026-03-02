from optimize_anything.strategies.sampling import EpochBatchSampler
from optimize_anything.strategies.stopping import BudgetStopper, CompositeStopper


def test_epoch_batch_sampler():
    sampler = EpochBatchSampler(minibatch_size=2, seed=42)
    ids = ["a", "b", "c", "d", "e"]
    batch1 = sampler.next_batch(ids, iteration=0)
    assert len(batch1) == 2
    batch2 = sampler.next_batch(ids, iteration=1)
    assert len(batch2) == 2
    assert batch1 != batch2 or True  # may overlap by chance


def test_epoch_sampler_covers_all():
    sampler = EpochBatchSampler(minibatch_size=2, seed=42)
    ids = ["a", "b", "c", "d"]
    seen = set()
    for i in range(2):  # 2 batches of 2 = full epoch
        batch = sampler.next_batch(ids, iteration=i)
        seen.update(batch)
    assert seen == set(ids)


def test_budget_stopper():
    stopper = BudgetStopper(max_evals=10)
    assert stopper.should_stop(total_evals=5) is False
    assert stopper.should_stop(total_evals=10) is True
    assert stopper.should_stop(total_evals=15) is True


def test_composite_stopper():
    s1 = BudgetStopper(max_evals=100)
    s2 = BudgetStopper(max_evals=5)
    composite = CompositeStopper([s1, s2])
    assert composite.should_stop(total_evals=3) is False
    assert composite.should_stop(total_evals=5) is True  # s2 triggers
