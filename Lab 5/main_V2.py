import math
import random
import numpy as np

# Simulate function (from the earlier implementation)
def simulate(force, x, x_dot, theta, theta_dot):
    GRAVITY = 9.8
    MASSCART = 1.0
    MASPOLE = 0.1
    TOTAL_MASS = MASPOLE + MASSCART
    LENGTH = 0.5
    POLEMASS_LENGTH = MASPOLE * LENGTH
    STEP = 0.02
    FOURTHIRDS = 4.0 / 3.0

    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (FOURTHIRDS - MASPOLE * costheta**2 / TOTAL_MASS))
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    # Euler integration for state update
    x += STEP * x_dot
    x_dot += STEP * xacc
    theta += STEP * theta_dot
    theta_dot += STEP * thetaacc

    return x, x_dot, theta, theta_dot


# Discretize state space
def discretize(x, x_dot, theta, theta_dot):
    # Define discretization bins for each variable
    x_bins = np.linspace(-2.4, 2.4, 10)
    x_dot_bins = np.linspace(-2.0, 2.0, 10)
    theta_bins = np.linspace(-12 * math.pi / 180, 12 * math.pi / 180, 10)
    theta_dot_bins = np.linspace(-2.0, 2.0, 10)

    # Discretize each variable
    x_discrete = np.digitize(x, x_bins)
    x_dot_discrete = np.digitize(x_dot, x_dot_bins)
    theta_discrete = np.digitize(theta, theta_bins)
    theta_dot_discrete = np.digitize(theta_dot, theta_dot_bins)

    return (x_discrete, x_dot_discrete, theta_discrete, theta_dot_discrete)


# Reward function
def reward(x, theta):
    if abs(x) > 2.4 or abs(theta) > 12 * math.pi / 180:
        return -1  # Failure
    return 0  # Survival


# Reinforcement learning with Q-learning
def train_agent():
    # Parameters
    actions = [-10, 10]  # Left and right forces
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Initial exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.01
    episodes = 1500

    # Initialize Q-table and visit count
    Q = {}  # Q-value table
    visit_count = {}  # Visit count for state-action pairs

    for episode in range(episodes):
        # Initialize state
        x, x_dot, theta, theta_dot = 0, 0, 0, 0
        state = discretize(x, x_dot, theta, theta_dot)

        # Episode loop
        for t in range(500):  # Limit the number of steps
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(actions)  # Explore
            else:
                action = max(actions, key=lambda a: Q.get((state, a), 0))  # Exploit

            # Simulate the action
            next_x, next_x_dot, next_theta, next_theta_dot = simulate(action, x, x_dot, theta, theta_dot)
            next_state = discretize(next_x, next_x_dot, next_theta, next_theta_dot)

            # Compute reward
            r = reward(next_x, next_theta)

            # Track visit count
            if (state, action) not in visit_count:
                visit_count[(state, action)] = 0
            visit_count[(state, action)] += 1

            # Compute dynamic learning rate
            alpha = 1 / visit_count[(state, action)]

            # Update Q-value using the Bellman equation with dynamic alpha
            best_next_action = max(actions, key=lambda a: Q.get((next_state, a), 0))
            Q[(state, action)] = Q.get((state, action), 0) + alpha * (r + gamma * Q.get((next_state, best_next_action), 0) - Q.get((state, action), 0))

            # Transition to the next state
            x, x_dot, theta, theta_dot = next_x, next_x_dot, next_theta, next_theta_dot
            state = next_state

            # End episode if failure occurs
            if abs(next_x) > 2.4 or abs(next_theta) > 12 * math.pi / 180:
                break

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, epsilon: {epsilon:.3f}")

    return Q


# Test the trained policy
def test_agent(Q):
    x, x_dot, theta, theta_dot = 0, 0, 0, 0
    state = discretize(x, x_dot, theta, theta_dot)

    print("\nTesting the agent: ")

    for t in range(500):  # Limit steps
        action = max([-10, 10], key=lambda a: Q.get((state, a), 0))
        x, x_dot, theta, theta_dot = simulate(action, x, x_dot, theta, theta_dot)
        state = discretize(x, x_dot, theta, theta_dot)

        print(f"Step {t}: x={x:.2f}, theta={theta:.2f}")
        if abs(x) > 2.4 or abs(theta) > 12 * math.pi / 180:
            print("System failed!")
            break
    else:
        print("System survived!")

#---------------------------------------------------------------------------#

Q_table = train_agent()

test_agent(Q_table)


