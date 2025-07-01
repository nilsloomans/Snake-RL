from snake_env import SnakeGame
from dqn_agent import DQNAgent
import time

# ------------------ Setup ------------------

NUM_GAMES = 5
DELAY = 0.1  # Sekunden zwischen Frames

env = SnakeGame()
agent = DQNAgent()
agent.load_model("model.pth")
agent.epsilon = 0  # Rein deterministisches Verhalten

# ------------------ Live-Spiel ------------------

for game_index in range(1, NUM_GAMES + 1):
    state = env.reset()
    done = False
    score = 0

    print(f"\n🎮 Spiel {game_index} startet...")

    while not done:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        score += reward
        env.render()
        time.sleep(DELAY)

    print(f"🏁 Spiel {game_index} beendet. Score: {score}")
    time.sleep(1)