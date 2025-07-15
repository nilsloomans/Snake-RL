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

## Q-Learning starten

```bash
jupyter notebook Q-Learning/qlearning_snake.ipynb
oder direkt im Notebook

## DQN starten:

```bash
jupyter notebook DQN/dqn_snake.ipynb
oder direkt im Notebook

## Informationen falls die Live-Demo am Ende nicht richtig funktioneirt (war manchmal der Fall) und sich das Spiel aufhängt/nicht mehr reagiert

Es ist schon ein fertiges Modell erstellt und im Q-Learning/DQN Ordner liegt jeweils eine play_trained.py.
Diese kann stattdessen ausgeführt werden um dieses Problem zu umgehen.
