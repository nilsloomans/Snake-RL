from snake_env import SnakeGame
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np

# ------------------ Einstellungen ------------------

NUM_EPISODES = 50
RENDER = False  # Auf True setzen für Live-Anzeige

# ------------------ Setup ------------------

env = SnakeGame()
agent = DQNAgent()
agent.load_model("model.pth")
agent.epsilon = 0  # Exploitation only (keine Zufallsaktionen)

scores = []

# ------------------ Evaluation ------------------

for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if RENDER:
            env.render()

    scores.append(total_reward)
    print(f"🏁 Episode {episode} beendet. Score: {total_reward}")

# ------------------ Ergebnisse ------------------

print("\n📊 Evaluation abgeschlossen")
print(f"🔢 Durchschnittlicher Score: {np.mean(scores):.2f}")
print(f"🏆 Maximaler Score: {np.max(scores)}")

# ------------------ Plot ------------------

plt.figure(figsize=(8, 5))
plt.hist(scores, bins=np.arange(0, max(scores)+2), edgecolor="black")
plt.title("Score-Verteilung (DQN Evaluation)")
plt.xlabel("Reward / Score")
plt.ylabel("Anzahl Episoden")
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_dqn_result.png")
plt.show()
