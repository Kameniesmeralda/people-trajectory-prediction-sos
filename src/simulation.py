from boid import Boid
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------
# 1- Paramètres de simulation

#Dimension de l'espace
width,height=100,100

#Nombre d'agents dans le système
N_boids= 30

#Paramètres comportementaux du modèle (règles + poids)
params={
    #poids de chaque force
    'w_separation':1.5,  # poids de la séparation
    'w_alignement':1.0,  #poids de l'alignement
    'w_cohesion':0.8,    # poids de la cohesion

    #rayons d'interaction
    'r_separation':15.0,   # rayon d'évitement
    'r_alignement':40.0,   # rayon d'alignement
    'r_cohesion':50.0,     # rayon de cohésion

    #Vitesse max
    'max_speed':3.0
}

#-----------------------------------------------
# 2- Initialisation des boids

#on crée une liste d'objets Boid avec des positions et des vitesses aléatoires
boids=[
    Boid(
        position=np.random.rand(2) * [width,height],   #positions aléatoires dans l'espace
        velocity=(np.random.rand(2)-0.5)*10            #vitesses initiales aléatoires
    )
    for _ in range(N_boids)
]

#-----------------------------------------------
#3-Configuration de l'affichage sur Matplotlib

plt.ion() # active le mode interactif (animation en direct)
fig,ax= plt.subplots(figsize=(7,7))
ax.set_xlim(0,width)
ax.set_ylim(0,height)
ax.set_title("Simulation of the Boids Model(Reynolds)")
ax.set_xlabel("X")
ax.set_ylabel("Y")

#On crée un nuage de points(scatter) qu'on mettra a jour à chaque itération
positions=np.array([b.pos for b in boids])
scat= ax.scatter(positions[:,0],positions[:,1], color="royalblue", s=30)

#-----------------------------------------------
#4-Boucle principale de simulation

for step in range(300): #nombre d'itérations
    #Mettre à jour chaque boid
    for b in boids:
        b.update(boids,params)  #appliquer les 3 règles comportementales
        b.apply_boundaries(width, height)
    #Mise à jour graphique
    positions=np.array([b.pos for b in boids]) #extraire les nouvelles positions
    scat.set_offsets(positions)                # mise à jour de la position du scatter
    ax.set_title(f"Simulation Boids - Itération {step+1}") # Titre dynamique

    plt.pause(0.03)  #pause courte pour l'animation

plt.ioff() #désactive le mode interactif
plt.show() #affiche la figure finale

#-----------------------------------------------
#5- Fin de la simulation

print("Simulation terminée")
