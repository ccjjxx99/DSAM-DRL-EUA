import cplex

from util.utils import get_reward


def can_allocate(workload, capacity):
    for i in range(4):
        if capacity[i] < workload[i]:
            return False
    return True


def cplex_allocate(original_servers, original_users, original_masks):
    servers = original_servers
    users = original_users
    m = len(servers)
    n = len(users)
    d = 4

    problem = cplex.Cplex()

    # 添加决策变量
    x = []
    for i in range(m):
        for j in range(n):
            x.append('x_{}_{}'.format(i, j))
    problem.variables.add(names=x, types=['B'] * m * n)

    y = []
    for i in range(m):
        y.append('y_{}'.format(i))
    problem.variables.add(names=y, types=['B'] * m)

    # 添加资源约束
    for i in range(m):
        for k in range(d):
            problem.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        ind=['x_{}_{}'.format(i, j) for j in range(n)] + ['y_{}'.format(i)],
                        val=[float(users[j][k + 2]) for j in range(n)] + [-float(servers[i][k + 3])]
                    )
                ],
                senses=['L'],
                rhs=[0]
            )

    # 添加覆盖约束
    for i in range(m):
        for j in range(n):
            problem.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        ind=['x_{}_{}'.format(i, j)],
                        val=[((users[j][0] - servers[i][0]) ** 2 + (users[j][1] - servers[i][1]) ** 2) ** 0.5]
                    )
                ],
                senses=['L'],
                rhs=[float(servers[i][2])]
            )

    # 添加分配约束
    for j in range(n):
        problem.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=['x_{}_{}'.format(i, j) for i in range(m)],
                    val=[1] * m
                )
            ],
            senses=['L'],
            rhs=[1]
        )

    # 添加目标函数
    obj1 = [(x[i], 1) for i in range(m * n)]
    obj2 = [(y[i], 1) for i in range(m)]
    problem.objective.set_sense(problem.objective.sense.maximize)
    problem.objective.set_linear(obj1)
    problem.objective.set_name('user_allocation')
    problem.multiobj.set_num(2)
    problem.multiobj.set_name(1, 'server_use')
    problem.multiobj.set_linear(1, obj2)
    problem.multiobj.set_weight(1, -1)
    problem.multiobj.set_priority(0, 2)
    problem.multiobj.set_priority(1, 1)
    problem.multiobj.set_abstol(0, 1)

    problem.set_log_stream(None)
    problem.set_error_stream(None)
    problem.set_warning_stream(None)
    problem.set_results_stream(None)

    # First, set the global deterministic time limit.
    problem.parameters.dettimelimit.set(600)

    # Second, create a parameter set for each priority.
    ps1 = problem.create_parameter_set()
    ps2 = problem.create_parameter_set()

    # Set the local deterministic time limits. Optimization will stop
    # whenever either the global or local limit is exceeded.
    ps1.add(problem.parameters.dettimelimit, 500)
    ps2.add(problem.parameters.dettimelimit, 250)

    # Optimize the multi-objective problem and apply the parameter
    # sets that were created above. The parameter sets are used
    # one-by-one by each optimization.
    problem.solve([ps1, ps2])

    problem.solve()

    solution = problem.solution

    actions = [-1] * n
    for j in range(n):
        for i in range(m):
            if solution.get_values('x_{}_{}'.format(i, j)) == 1:
                actions[j] = i

    user_allocate_list, server_allocate_num, user_allocated_prop, server_used_prop, capacity_used_prop = \
        get_reward(original_servers, original_users, actions)

    return None, None, user_allocate_list, server_allocate_num, \
        user_allocated_prop, server_used_prop, capacity_used_prop
