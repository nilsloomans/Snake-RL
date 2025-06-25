import numpy as np
import random
from collections import defaultdict
import pickle

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.actions = actions  # [0, 1, 2] → [straight, left, right]
        self.alpha = alpha      # Lernrate
        self.gamma = gamma      # Discount-Faktor
        self.epsilon = epsilon  # Explorationsrate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-Tabelle: Schlüssel = (state tuple), value = dict{action: value}
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def get_state_key(self, state):
        # Um aus numpy array einen hashbaren key zu machen
        return tuple(state.tolist())

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if random.uniform(0, 1) < self.epsilon:
            # Zufällige Aktion (Exploration)
            return random.choice(self.actions)
        else:
            # Beste bekannte Aktion (Exploitation)
            return int(np.argmax(self.q_table[state_key]))

    def update(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_key]) if not done else 0

        # Q-Learning Formel
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state_key][action] = new_value

        # Epsilon anpassen
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_table(self):
        return dict(self.q_table)  # für Export/Analyse
    
    def save(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
            self.q_table.update(data)
