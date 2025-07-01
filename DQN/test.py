from dqn_agent import DQNAgent
import numpy as np

agent = DQNAgent()
state = np.zeros(11, dtype=int)

action = agent.choose_action(state)
print("Gewählte Aktion:", action)

agent.remember(state, action, reward=1.0, next_state=state, done=False)
agent.train_step()

print("Trainingsschritt durchgeführt.")
