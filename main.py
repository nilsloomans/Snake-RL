from snake_env import SnakeGame

game = SnakeGame()

state = game.reset()
print("initialer Zusstand:", state)

for _ in range(10):
    state, reward, done, _ = game.step(0)
    print(f"State: {state.tolist()}, Reward: {reward}, Done: {done}")
    game.render()
    if done:
        break