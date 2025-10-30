import numpy as np

class Boid:

    """ Classe Boid: représente un agent qui sera une personne dans notre cas"""

    def __init__(self,position,velocity):
        self.pos=np.array(position)
        self.vel=np.array(velocity)
        self.best_pos= np.copy(self.pos) #Meilleure position personnelle de l'agent

    def evaluate(self, objective_fitness):
        """ Evalue la qualité de la position actuelle et met à jour
        la meilleure position personnelle si nécessaire"""

        current_fitness= objective_fitness(self.pos)
        best_fitness= objective_fitness(self.best_pos)
        if current_fitness<best_fitness:
            self.best_pos=np.copy(self.pos)

    def limit_speed(self, max_speed):
        """ Limite la norme de la vitesse à une valeur maximale"""
        speed = np.linalg.norm(self.vel)
        if speed > max_speed:
            self.vel = (self.vel / speed) * max_speed

    # Méthode de mise à jour

    def update(self,global_best_pos,params):
        """ Mets à jour la vitesse et la position du Boid à chaque pas de temps
        selon la formule de l'algorithme PSO
        """

        # Extraction des paramètres
        w = params.get('w_inertia',0.7) #inertie
        c1 = params.get('c1', 1.5) #influence personnel
        c2 = params.get('c2', 1.5) #influence social
        max_speed = params.get('max_speed', 3.0) #séparation

        # les facteurs aléatoires
        r1 = np.random.rand()
        r2 = np.random.rand()

        # Mise à jour de la vitesse

        personal_influence= c1*r1*(self.best_pos - self.pos)
        social_influence=c2*r2*(global_best_pos - self.pos)
        inertia_component= w*self.vel

        self.vel= inertia_component + personal_influence + social_influence

        #Limiter la vitesse pour éviter les explosions numériques
        self.limit_speed(max_speed)

        #Mise à jour de la position
        self.pos+= self.vel

