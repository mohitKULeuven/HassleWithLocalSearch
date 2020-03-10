from typing import Optional

import numpy as np
from pysdd.iterator import SddIterator
from pysdd.sdd import Vtree, SddManager

from pysat_solver import solve_weighted_max_sat, get_value
from type_def import MaxSatModel, Context, Clause


def find_solutions_rec(weights, selected, available, budget):
    if budget == 0:
        yield selected

    for i in available:
        s = selected | {i}
        b = budget - weights[i]
        a = {j for j in available if j > i and weights[j] <= b}
        yield from find_solutions_rec(weights, s, a, b)


def find_weight_assignments(weights, budget):
    weights = np.array(weights)
    selected = set()
    available = {i for i in range(len(weights)) if weights[i] <= budget}

    return find_solutions_rec(weights, selected, available, budget)


def clause_to_sdd(clause: Clause, manager: SddManager):
    result = manager.false()
    for i in clause:
        if i > 0:
            result |= manager.literal(i)
        else:
            result |= ~manager.literal(abs(i))
    return result


def get_sdd_manager(n: int):
    vtree = Vtree(n, list(range(1, n+1)), "balanced")
    return SddManager.from_vtree(vtree)


def convert_to_logic(manager: SddManager, n: int, model: MaxSatModel, context: Context):
    best_solution, value = solve_weighted_max_sat(n, model, context, 1)

    if value is -1:
        return None
    else:
        value = get_value(model, best_solution, context)
        hard_constraints = [c for w, c in model if not w]
        soft_constraints = [(w, c) for w, c in model if w is not None]
        weights = [t[0] for t in soft_constraints]
        assignments = find_weight_assignments(weights, value)

        hard_result = manager.true()
        for clause in hard_constraints:
            hard_result &= clause_to_sdd(clause, manager)

        soft_result = manager.false()
        for assignment in assignments:
            assignment_result = manager.true()
            for i in assignment:
                assignment_result &= clause_to_sdd(soft_constraints[i][1], manager)
            soft_result |= assignment_result

        return hard_result & soft_result


def count_solutions(n: int, model: MaxSatModel, context: Context):
    # TODO What we actually want: optimal across all possible contexts!

    manager = get_sdd_manager(n)
    logic = convert_to_logic(manager, n, model, context)
    if logic is None:
        return 0

    return logic.global_model_count()


def get_recall_precision(n: int, true_model: MaxSatModel, learned_model: MaxSatModel, context: Context):
    manager = get_sdd_manager(n)
    true_logic = convert_to_logic(manager, n, true_model, context)
    learned_logic = convert_to_logic(manager, n, learned_model, context)
    combined = true_logic & learned_logic

    true_count, learned_count, combined_count = (l.global_model_count() for l in (true_logic, learned_logic, combined))
    return combined_count / true_count, combined_count / learned_count


def simple():
    n = 2
    true_model = [(1, {1}), (1, {2})]
    learned_model = [(5, {1})]
    r, p = get_recall_precision(n, true_model, learned_model, set())
    print(r, p)


if __name__ == '__main__':
    simple()
