import numpy as np
import logging
import pickle
import os

from sample_models import generate_model
from pysat_solver import solve_weighted_max_sat, label_instance
from type_def import MaxSatModel, Context

logger = logging.getLogger(__name__)


def generate_models(n, max_clause_length, num_hard, num_soft, model_seed):
    param = f"_n_{n}_max_clause_length_{max_clause_length}_num_hard_{num_hard}_num_soft_{num_soft}_model_seed_{model_seed}"
    if os.path.exists("pickles/target_model/" + param + ".pickle"):
        true_model = pickle.load(
            open("pickles/target_model/" + param + ".pickle", "rb")
        )["true_model"]
        return true_model, param
    rng = np.random.RandomState(model_seed)
    true_model = generate_model(n, max_clause_length, num_hard, num_soft, rng)
    pickle_var = {}
    pickle_var["true_model"] = true_model
    pickle.dump(pickle_var, open("pickles/target_model/" + param + ".pickle", "wb"))
    return true_model, param


def generate_contexts_and_data(
    n, model, num_context, num_pos, num_neg, param, context_seed
):
    param += f"_num_context_{num_context}_num_pos_{num_pos}_num_neg_{num_neg}_context_seed_{context_seed}"
    if os.path.exists("pickles/contexts_and_data/" + param + ".pickle"):
        return param
    pickle_var = {}
    rng = np.random.RandomState(context_seed)
    pickle_var["contexts"] = []
    pickle_var["data"] = []
    pickle_var["labels"] = []
    if num_context == 0:
        data, labels = generate_data(n, model, set(), num_pos, num_neg, context_seed)
        pickle_var["contexts"].extend([set()] * len(data))
        pickle_var["data"].extend(data)
        pickle_var["labels"].extend(labels)
    else:
        for _ in range(num_context):
            context, data_seed = random_context(n, rng)
            data, labels = generate_data(n, model, context, num_pos, num_neg, data_seed)
            pickle_var["contexts"].extend([context] * len(data))
            pickle_var["data"].extend(data)
            pickle_var["labels"].extend(labels)

    pickle.dump(
        pickle_var, open("pickles/contexts_and_data/" + param + ".pickle", "wb")
    )
    return param


def random_context(n, rng):
    clause = []
    indices = rng.choice(range(n), 2, replace=False)
    for i in range(n):
        if i in indices:
            clause.append(rng.choice([-1, 1]))
        else:
            clause.append(0)
    context = []
    for j, literal in enumerate(clause):
        if literal != 0:
            context.append((j + 1) * literal)
    data_seed = rng.randint(1, 1000)
    return context, data_seed


# def random_context(n, rng):
#    random_index = rng.randint(1, pow(3, n))
#    clause = ternary(random_index, n)
#    clause = [-1 if j == 2 else j for j in clause]
#
#    context = []
#    for j, literal in enumerate(clause):
#        if literal != 0:
#            context.append((j + 1) * literal)
#    data_seed = rng.randint(1, 1000)
#    return context, data_seed


def generate_data(n, model: MaxSatModel, context: Context, num_pos, num_neg, seed):
    rng = np.random.RandomState(seed)
    labels = []
    data = []
    tmp_data, cst = solve_weighted_max_sat(n, model, context, num_pos * 10)
    if len(tmp_data) > num_pos:
        indices = list(rng.choice(range(len(tmp_data)), num_pos, replace=False))
        for i in indices:
            data.append(tmp_data[i])
    else:
        data = tmp_data
    num_pos = len(data)
    labels = [True] * num_pos
    max_tries = 1000 * num_neg
    for l in range(max_tries):
        instance = rng.rand(n) > 0.5
        for i in rng.choice(list(context), 1):
            instance[abs(i) - 1] = i > 0
        if list(instance) in data:
            continue
        if not label_instance(model, instance, context):
            data.append(list(instance))
            labels.append(False)
            if len(data) >= num_neg + num_pos:
                break
    return data, labels
