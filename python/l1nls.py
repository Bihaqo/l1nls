import pulp


def l1nls(A, b):
    """Find the solution of system of linear equations with minimal 1-norm.

    Solves an optimization problem
    min. ||x||_1
    s.t.
    A x = b
    where ||x||_1 is the 1-norm of the vector x.
    """
    [m, n] = A.shape
    prob = pulp.LpProblem("Linear System", pulp.LpMinimize)
    pos_vars = pulp.LpVariable.dicts("Positive", range(n), 0)
    neg_vars = pulp.LpVariable.dicts("Negative", range(n), 0)
    prob += pulp.lpSum(pos_vars[i] for i in range(n)) + pulp.lpSum(neg_vars[i] for i in range(n))
    for eq in range(m):
        prob += pulp.lpSum((pos_vars[i] - neg_vars[i]) * A[eq][i] for i in range(n)) == b[eq]

    prob.solve()
    x = [pulp.value(pos_vars[i]) - pulp.value(neg_vars[i]) for i in range(n)]
    val = pulp.value(prob.objective)
    return (x, val)
