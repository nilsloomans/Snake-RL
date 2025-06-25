from snake_env import SnakeGame
from agent import QLearningAgent
import time

# Umgebung & Agent
env = SnakeGame()
agent = QLearningAgent(actions=[0, 1, 2])
agent.load("q_table.pkl")  # Q-Tabelle aus Training laden
agent.epsilon = 0          # nur Exploitation (kein Zufall)

# Anzahl Live-Spiele
NUM_GAMES = 3

for i in range(1, NUM_GAMES + 1):
    state = env.reset()
    done = False
    score = 0

    print(f"\n🎮 Starte Spiel {i}...")

    while not done:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        score += reward
        env.render()
        time.sleep(0.1)  # Langsamer anzeigen

    print(f"🏁 Spiel {i} beendet. Score: {score}")
    time.sleep(1)
