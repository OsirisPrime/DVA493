import numpy as np
import random
import matplotlib.pyplot as plt

force_values = [-10, 10]  # forces to apply (N)

def simulate_pendulum(state, force):
    x, x_dot, theta, theta_dot = state

    # Constants
    g = -9.8  # acceleration due to gravity (m/s^2)
    mc = 1.0  # mass of the cart (kg)
    m = 0.1  # mass of the pole (kg)
    l = 0.5  # half of the pole length (m)
    dt = 0.02  # time interval (s)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    total_mass = mc + m
    pole_mass_length = m * l

    temp = (force + pole_mass_length * theta_dot ** 2 * sin_theta) / total_mass

    theta_acc = (g * sin_theta - cos_theta * temp) / \
                (l * (4.0 / 3.0 - m * cos_theta ** 2 / total_mass))

    x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass

    # Update state using Euler's method
    x += dt * x_dot
    x_dot += dt * x_acc
    theta += dt * theta_dot
    theta_dot += dt * theta_acc

    return np.array([x, x_dot, theta, theta_dot])

def discretize_state(state):
    buckets = [
        np.linspace(-2.4, 2.4, 5),  # cart position
        np.linspace(-3.0, 3.0, 5),  # cart velocity
        np.linspace(-0.209, 0.209, 5),  # pole angle (radians ~12 degrees)
        np.linspace(-2.0, 2.0, 5)  # pole angular velocity
    ]

    indices = []
    for i, val in enumerate(state):
        indices.append(np.digitize(val, buckets[i]) - 1)

    return tuple(indices)

# Reward function
def reward_function(state):
    x, _, theta, _ = state
    if abs(x) > 2.4 or abs(theta) > 0.209:  # Out of bounds
        return -1
    else:
        return 1

def train_model():
    # Initialize Q-table
    state_space = (5, 5, 5, 5)  # 10 buckets per state variable
    q_table = np.zeros(state_space + (len(force_values),))

    # Hyperparameters
    initial_alpha = 0.5  # Initial learning rate
    min_alpha = 0.01  # Minimum learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    min_epsilon = 0.01

    episodes = 1000
    max_steps = 3000

    for episode in range(episodes):
        state = np.array([0, 0, 0, 0])  # Initial state
        discretized_state = discretize_state(state)
        total_reward = 0

        # Dynamically update alpha based on the episode
        alpha = max(min_alpha, initial_alpha * (0.85 ** (episode // 100)))

        for step in range(max_steps):
            if random.random() < epsilon:
                action = random.choice([0, 1])  # Explore
            else:
                action = np.argmax(q_table[discretized_state])  # Exploit

            force = force_values[action]

            next_state = simulate_pendulum(state, force)
            reward = reward_function(next_state)
            next_discretized_state = discretize_state(next_state)

            total_reward += reward

            # Update Q-value
            best_future_q = np.max(q_table[next_discretized_state])
            q_table[discretized_state + (action,)] += alpha * (
                reward + gamma * best_future_q - q_table[discretized_state + (action,)]
            )

            if reward == -1:  # Terminate if out of bounds
                break

            state = next_state
            discretized_state = next_discretized_state

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    return q_table

def test_model(q_table):
    state = np.array([0, 0, 0, 0])  # Initial state
    discretized_state = discretize_state(state)

    positions = []
    angles = []

    for step in range(3000):  # Test for 3000 steps
        positions.append(state[0])
        angles.append(np.degrees(state[2]))

        action = np.argmax(q_table[discretized_state])  # Always exploit during testing
        force = force_values[action]

        state = simulate_pendulum(state, force)
        if reward_function(state) == -1:
            print(f"Test failed: System lasted {step} steps before going out of bounds.")
            break

        discretized_state = discretize_state(state)
    else:
        print("Test passed: System survived 3000 steps.")

    # Plotting results
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # Two rows, one column

    # Plot position (x)
    axes[0].plot(positions, label="Position (x)")
    axes[0].set_ylim(-2.4, 2.4)
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Position (x)")
    axes[0].set_title("Position (x) over Time")
    axes[0].legend()
    axes[0].grid()

    # Plot angle (theta)
    axes[1].plot(angles, label="Angle (theta)", color="orange")
    axes[1].set_ylim(-12, 12)
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Angle (theta) [degree]")
    axes[1].set_title("Angle (theta) over Time")
    axes[1].legend()
    axes[1].grid()


    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------------#

q_table = train_model()
test_model(q_table)

