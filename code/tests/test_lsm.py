from code.generator import sample_models, random_data, random_context
import numpy as np
import pickle
from code.pysat_solver import get_value


def test_model_generator():
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


def test_data_generator():
    rng = np.random.RandomState(111)

    for model in sample_models(
        num_models=10, num_vars=5, clause_length=2, num_hard=5, num_soft=5, rng=rng
    ):
        for _ in range(5):
            context, data_seed = random_context(5, rng)
            data1, labels1 = random_data(
                n=5, model=model, context=context, num_pos=1, num_neg=2, seed=data_seed
            )
            assert len(data1) <= 3
            assert len(data1) == len(labels1)

            data2, labels2 = random_data(
                n=5, model=model, context=context, num_pos=1, num_neg=10, seed=data_seed
            )
            assert len(data2) <= 11
            assert len(data2) == len(labels2)

            for i, d in enumerate(data1):
                if not labels1[i]:
                    assert data2[i] == data1[i]


def test_model_copy():
    model = [(None, {1, 2, 3}), (0.5, {2, 4})]
    model2 = [(None, {1, 2, 3}), (0.5, {2, 4})]
    instance = np.zeros(5)
    context = {3, 4, 5}
    get_value(model, instance, context)
    assert model == model2
