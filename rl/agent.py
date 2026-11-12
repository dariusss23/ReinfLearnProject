"""Agentul de invatare (algoritmul inca nu a fost ales: PPO, DQN etc.).

Aici va intra tot ce tine de algoritm: retelele, bufferul de experienta si
regula de update.

Interfata minima (de respectat, ca restul codului sa nu depinda de algoritm):
    agent.act(obs, mask)  -> actiunea aleasa (+ date extra daca algoritmul le cere)
    agent.update(batch)   -> dict cu metrici
"""
