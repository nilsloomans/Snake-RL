# Snake Reinforcement Learning – Q-Learning & DQN

Dieses Projekt demonstriert die Anwendung von **Reinforcement Learning** auf das klassische Spiel **Snake**. Dabei werden zwei unterschiedliche Agenten verglichen:

- **Q-Learning** (tabellarisch)
- **DQN (Deep Q-Network)** mit neuronalen Netzen und erweitertem Zustandsraum

Ziel ist es, eine möglichst effektive Strategie zu entwickeln, mit der die Schlange durch das Spielfeld navigiert, Äpfel frisst und dabei nicht mit sich selbst oder den Wänden kollidiert.

## Was wurde gemacht?

- Implementierung einer Snake-Umgebung mit `pygame`
- Entwicklung eines **Q-Learning-Agenten** mit diskreter Zustandsdarstellung
- Entwicklung eines **DQN-Agenten** mit neuronaler Netzarchitektur, Replay Buffer und Flood-Fill-Feature
- Vergleich der beiden Ansätze hinsichtlich Lernverlauf, Stabilität und Performance
- Visualisierung der Ergebnisse inkl. Score-Histogrammen und Trainingskurven
- Live-Demonstration des Agentenverhaltens im gerenderten Spiel
- Komplette Umsetzung in **Jupyter Notebooks**

---

## Ausführung

## Q-Learning:
```bash
jupyter notebook Q-Learning/qlearning_snake.ipynb
oder direkt im Notebook
