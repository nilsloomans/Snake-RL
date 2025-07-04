from snake_env import SnakeGame
from dqn_agent import DQNAgent
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

# -------------------- Parameter --------------------

EPISODES = 10000
MAX_STEPS = 500
RENDER = False
PLOT_EVERY = 200  # Plot-Ausgabe nach x Episoden

# -------------------- Setup --------------------

env = SnakeGame()
agent = DQNAgent()
scores = []

# -------------------- Training --------------------

for episode in trange(EPISODES, desc="Training"):
    state = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        if RENDER:
            env.render()

        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state
        total_reward += reward

        if done:
            break

    scores.append(total_reward)

    if (episode + 1) % PLOT_EVERY == 0:
        avg = np.mean(scores[-PLOT_EVERY:])
        print(f"📈 Episode {episode+1}, Ø Reward letzte {PLOT_EVERY} Episoden: {avg:.2f}")

# -------------------- Ergebnis-Plot --------------------

def moving_average(data, window=100):
    return np.convolve(data, np.ones(window) / window, mode="valid")

plt.figure(figsize=(10, 5))
plt.plot(scores, label="Reward pro Episode")
plt.plot(moving_average(scores), label="Gleitender Durchschnitt (100)", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Trainingsverlauf (DQN mit Target Network)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_dqn_result.png")
plt.show()

# -------------------- Modell speichern --------------------

agent.save_model("model.pth")
print("✅ Modell gespeichert als model.pth")