from code.generator import sample_models
import numpy as np


def test_generator():
    rng = np.random.RandomState(111)

    for model in sample_models(
        num_models=10, num_vars=10, clause_length=3, num_hard=9, num_soft=11, rng=rng
    ):
        assert len(model) == 20
        h, s = 0, 0
        for w, clause in model:
            assert len(clause) <= 3
            if w:
                assert w <= 1
                assert w > 0
                s += 1
            else:
                h += 1
            for c in clause:
                assert c <= 10
        assert h == 9
        assert s == 11
