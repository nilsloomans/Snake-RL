from snake_env import SnakeGame
from agent import QLearningAgent
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np

# ------------------ Parameter ------------------

EPISODES = 1000
MAX_STEPS = 500
NUM_EVAL_EPISODES = 50
MOVING_AVG_WINDOW = 20

# ------------------ Initialisierung ------------------

env = SnakeGame()
agent = QLearningAgent(actions=[0, 1, 2])  # 0=straight, 1=left, 2=right

scores = []

# ------------------ Training ------------------

print("Starte Training...\n")

for episode in trange(EPISODES, desc="Training"):
    state = env.reset()
    total_reward = 0

    for _ in range(MAX_STEPS):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    scores.append(total_reward)

# ------------------ Gleitender Durchschnitt ------------------

def moving_average(data, window=MOVING_AVG_WINDOW):
    return np.convolve(data, np.ones(window)/window, mode='valid')

smoothed_scores = moving_average(scores)

# ------------------ Plot: Trainingsverlauf ------------------

plt.figure()
plt.plot(scores, alpha=0.3, label="Original")
plt.plot(range(len(smoothed_scores)), smoothed_scores, color='orange', label=f"Moving Average (window={MOVING_AVG_WINDOW})")
plt.title("Reward pro Episode (Trainingsphase)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_result.png")
plt.show()

# ------------------ Evaluation ------------------

print("\nStarte Evaluation des trainierten Agenten...")

agent.epsilon = 0  # Exploration deaktivieren
eval_scores = []

for _ in range(NUM_EVAL_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    eval_scores.append(total_reward)

# ------------------ Plot: Evaluation ------------------

plt.figure()
plt.hist(eval_scores, bins=np.arange(0, max(eval_scores)+2, 1), edgecolor='black')
plt.title("Verteilung der Scores im Testmodus (ε = 0)")
plt.xlabel("Reward (Score)")
plt.ylabel("Häufigkeit")
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_result.png")
plt.show()

# ------------------ Ergebnis ------------------

print(f"Durchschnittlicher Score über {NUM_EVAL_EPISODES} Testspiele: {np.mean(eval_scores):.2f}")
print(f"Maximaler Score im Test: {np.max(eval_scores)}")

# ------------------ Speichern ------------------

agent.save("q_table.pkl")
print("Q-Tabelle gespeichert als q_table.pkl ✅")