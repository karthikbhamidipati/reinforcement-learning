import numpy as np


def calc_prob_rewards(env):
    p = np.empty([env.n_states, env.n_states, env.n_actions])
    r = np.empty_like(p)
    for s in range(env.n_states):
        for s_prime in range(env.n_states):
            for action in range(env.n_actions):
                p[s, s_prime, action] = env.p(s_prime, s, action)
                r[s, s_prime, action] = env.r(s_prime, s, action)
    return p, r


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    curr_iteration = 0
    stop = False

    p, r = calc_prob_rewards(env)

    while curr_iteration < max_iterations and not stop:
        delta = 0

        for s in range(env.n_states):
            current_value = value[s]

            # for action in range(env.n_actions):
            #     for s_prime in range(env.n_states):
            #         value += policy[s] * \
            #                  p[s, s_prime, action] * \
            #                  (r[s, s_prime, action] + (gamma * value[s_prime]))

            value[s] = policy[s] * np.sum(p[s] * (r[s] + (gamma * value.reshape(-1, 1))))
            delta = max(delta, abs(current_value - value[s]))

        curr_iteration += 1
        stop = delta >= theta

    return value


def policy_improvement(env, policy, value, gamma):
    improved_policy = np.zeros(env.n_states, dtype=int)

    # TODO ADD logic

    return improved_policy


def policy_iteration(env, gamma, theta, max_iterations):
    policy = np.zeros(env.n_states, dtype=int)
    value = np.zeros(env.n_states, dtype=np.float)

    # TODO ADD logic

    return policy, value


def value_iteration(env, gamma, theta, max_iterations):
    policy = np.zeros(env.n_states, dtype=int)
    value = np.zeros(env.n_states, dtype=np.float)

    # TODO ADD logic

    return policy, value
