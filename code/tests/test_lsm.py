from code.generator import sample_models


def test_generator():
    models = sample_models(
        num_models=10, num_vars=5, clause_length=3, num_hard=9, num_soft=11
    )
    assert len(models) == 10
    for model in models:
        assert len(model) == 20
        h, s = 0, 0
        for w, clause in model:
            assert len(clause) == 3
            if w:
                assert w <= 1
                assert w > 0
                s += 1
            else:
                h += 1
            for c in clause:
                assert c <= 5
        assert h == 9
        assert s == 11
