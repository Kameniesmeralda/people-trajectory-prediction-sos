from boid import Boid
import numpy as np
import matplotlib.pyplot as plt

# exemple de fonction fitness

def objective_fitness(position):
    #Exemple: il faut atteindre là position [50,50]

    return np.linalg.norm(position-np.array([50,50]))

# Initialisation de la population

N= 50

""" On initialise une liste de 20 personnes réparties aléatoirement 
dans un espace 2D de 100*100 avec chacune une position et une vélocité aléatoires"""

boids= [Boid(np.random.rand(2)*100, np.random.randn(2)) for _ in range(N)]

# Paramètres de notre algorithme PSO

params= {'w_inertia':0.7, 'c1':2.0, 'c2':2.0, 'max_speed':3.0}

#Initialisation du meilleur global
global_best_pos= boids[0].pos.copy()
global_best_value=objective_fitness(global_best_pos)

# affichage de la simulation
plt.ion() # mode interactif avec animation
fig, ax= plt.subplots(figsize=(6,6))

#Boucle de simulation
for t in range(100):
    #évaluation et mise à jour du meilleur global
    for b in boids:
        b.evaluate(objective_fitness)
        # Mise à jour du meilleur global
        if objective_fitness(b.best_pos)<global_best_value:
            global_best_pos=b.best_pos.copy()
            global_best_value=objective_fitness(b.best_pos)

    #mise à jour des boids
    for b in boids:
        b.update(global_best_pos, params)

    # Visualisation
    ax.clear()
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.set_title(f"Iteration {t+1} - Best value: {global_best_value:4e}")

    # Position des agents
    positions= np.array([b.pos for b in boids])
    ax.scatter(positions[:,0],positions[:,1],color='blue', label='Boids')

    #Meilleure position globale
    ax.scatter(global_best_pos[0], global_best_pos[1], color='red', marker='*', s=200, label='Best Global')

    #Point cible
    ax.scatter(50,50, color='green', marker='X',s=100, label="Target(50,50)")

    ax.legend()
    plt.pause(0.005)

plt.ioff()
plt.show()

# Résultats finaux
print("Best global position found:", global_best_pos)
print("Objective value:",global_best_value)
