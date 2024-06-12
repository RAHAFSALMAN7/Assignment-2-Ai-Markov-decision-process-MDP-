 # Define the MDP components
states = ['S1', 'S2', 'S3']
actions = ['a1', 'a2', 'a3']
transition_probabilities = {
    'S1': {
        'a1': {'S1': 0.001, 'S2': 0.999},
        'a2': {'S1': 0.001, 'S3': 0.999}
    },
    'S2': {
        'a3': {'S1': 0.001, 'S3': 0.999}
    },
    'S3': {}
}
rewards = {
    'S1': {'a1': 100, 'a2': 90},
    'S2': {'a3': 11}
}

def initialize_values(states):
    return {state: 0 for state in states}

def value_iteration_fixed_iterations(V, gamma, iterations):
    history = []
    for i in range(iterations):
        new_V = V.copy()
        for s in states:
            if s in transition_probabilities and transition_probabilities[s]:
                new_V[s] = min(
                    sum(transition_probabilities[s][a][s_next] * (rewards[s].get(a, 0) + gamma * V[s_next])
                        for s_next in transition_probabilities[s][a])
                    for a in transition_probabilities[s]
                )
        V = new_V
        history.append(V.copy())
    return V, history

# Run value iteration without discounting (gamma = 1) for a fixed number of iterations
print("Without discount factor:")
V_no_discount, _ = value_iteration_fixed_iterations(initialize_values(states), gamma=1, iterations=4)
for i, values in enumerate(_):
    print(f"Iteration {i + 1}: {values}")

# Run value iteration with discounting (gamma = 0.9) for a fixed number of iterations
print("\nWith discount factor (gamma = 0.9):")
V_with_discount, _ = value_iteration_fixed_iterations(initialize_values(states), gamma=0.9, iterations=4)
for i, values in enumerate(_):
    print(f"Iteration {i + 1}: {values}")

# Extract policy
def extract_policy(V, gamma):
    policy = {}
    for s in states:
        if s in transition_probabilities and transition_probabilities[s]:
            policy[s] = min(
                (a for a in transition_probabilities[s]),
                key=lambda a: sum(transition_probabilities[s][a][s_next] * (rewards[s].get(a, 0) + gamma * V[s_next])
                                  for s_next in transition_probabilities[s][a])
            )
        else:
            policy[s] = None
    return policy

policy_no_discount = extract_policy(V_no_discount, gamma=1)
print("\nOptimal policy without discounting:", policy_no_discount)

policy_with_discount = extract_policy(V_with_discount, gamma=0.9)
print("Optimal policy with discounting:", policy_with_discount)
