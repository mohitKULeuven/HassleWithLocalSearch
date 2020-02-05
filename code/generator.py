import numpy as np
import logging
import pickle

# import os

from sample_models import generate_model
from pysat_solver import solve_weighted_max_sat, label_instance
from type_def import MaxSatModel, Context
from local_search import ternary


logger = logging.getLogger(__name__)


def generate_models(n, max_clause_length, num_hard, num_soft, model_seed):
    rng = np.random.RandomState(model_seed)
    pickle_var = {}
    true_model = generate_model(n, max_clause_length, num_hard, num_soft, rng)
    pickle_var["true_model"] = true_model
    param = f"_n_{n}_max_clause_length_{max_clause_length}_num_hard_{num_hard}_num_soft_{num_soft}_model_seed_{model_seed}"
    pickle.dump(pickle_var, open("pickles/target_model" + param + ".pickle", "wb"))
    return true_model, param


def generate_contexts_and_data(n, model, num_context, num_data, param, context_seed):
    pickle_var = {}
    rng = np.random.RandomState(context_seed)
    pickle_var["contexts"] = []
    pickle_var["data"] = []
    pickle_var["labels"] = []
    pickle_var["learning_contexts"] = []
    for _ in range(num_context):
        context, data_seed = random_context(n, rng)
        data, labels = generate_data(n, model, context, num_data, data_seed)
        pickle_var["contexts"].extend([context] * len(data))
        pickle_var["data"].extend(data)
        pickle_var["labels"].extend(labels)
    param += (
        f"_num_context_{num_context}_num_data_{num_data}_context_seed_{context_seed}"
    )
    pickle.dump(pickle_var, open("pickles/contexts_and_data" + param + ".pickle", "wb"))
    return param


def random_context(n, rng):
    random_index = rng.randint(1, pow(3, n))
    clause = ternary(random_index, n)
    clause = [-1 if j == 2 else j for j in clause]

    context = []
    for j, literal in enumerate(clause):
        if literal != 0:
            context.append((j + 1) * literal)
    data_seed = rng.randint(1, 1000)
    return context, data_seed


def generate_data(n, model: MaxSatModel, context: Context, num_data, seed):
    rng = np.random.RandomState(seed)
    labels = []
    data, cst = solve_weighted_max_sat(n, model, context, num_data)
    labels = [True] * len(data)
    #    output = np.array(output).astype(np.int)
    max_tries = 100 * num_data
    for l in range(max_tries):
        instance = rng.rand(n) > 0.5
        for i in context:
            instance[abs(i) - 1] = i > 0
        #        print(model, instance, context)
        if not label_instance(model, instance, context):
            data.append(list(instance))
            labels.append(False)
            if len(data) >= num_data:
                break
    #    print(data)
    return data, labels
