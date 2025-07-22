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

bash
jupyter notebook Q-Learning/Q-Learning.ipynb
oder direkt im Notebook (ist empfohlen)

## DQN:

bash
jupyter notebook DQN/DQN.ipynb
oder direkt im Notebook (ist empfohlen)

## Weitere Informationen zur Ausführung

Es kann sein, dass die Live-Demo am Ende nicht immer funktioniert.
Das pygame hängt sich manchmal auf, vorallem wenn man das Fenster anklickt/verschiebt.
Neben dem Notebook liegt der Code auch in einzelnen Python-Dateien vor.
In jedem der 2 Ordner für Q-Learning und DQN liegt jeweils eine Datei mit play_trained.py.
Diese können stattdessen ausgeführt werden um die Agenten entsprechend zu nutzen.
Es liegen nämlich 2 Modelle jeweils 1 zu jedem Ansatz schon da.

## Vorausetzungen

- Der Code wurde mit der Python Version 3.12.1 ausgeführt
- Die verschiedenen benötigten Packages sind in der requirements dokumentiert
