import numpy as np

class Boid:

    """ Classe Boid : représente un agent qui sera une personne dans notre cas
    qui se déplace dans un espace 2D en suivant les trois règles de Reynolds
    – séparation : éviter les collisions avec les voisins proches
    — alignement : ajuster sa direction en fonction de la moyenne du groupe
    — cohésion : se rapprocher du centre du groupe

    """

    def __init__(self,position,velocity):

        """" Initialise un boid avec une position et une vitesse initiales
        position : (x,y) initiale du boid
        velocity : vecteur vitesse initial (vx,vy)
        """

        self.pos=np.array(position)
        self.vel=np.array(velocity)


    # ---------------------------------------------------------------------
    # Règle de Séparation : éviter de trop se rapprocher des voisins

    def separation (self, boids, params):

        """"Calcule un vecteur qui repousse le boid de ses voisins trop proches"""

        steer= np.zeros(2)  # vecteur de force initialisé à 0
        r_sep= params['r_separation'] #rayon de séparation défini dans les paramètres"

        for other in boids:
            if other is self: # on parcourt tous les boids si c'est soi-même alors on ignore
                continue
            offset=self.pos-other.pos #on trouve le vecteur de distance d'un autre boid à soi
            distance= np.linalg.norm(offset) #distance entre les deux boids
            if r_sep > distance > 0: #si l'autre boid est dans le rayon de séparation
                steer+=offset/distance # on ajoute une force inversement proportionnelle à la distance

        return steer # force totale de séparation


    # ---------------------------------------------------------------------
    # Règle d'alignement : suivre la direction moyenne du groupe

    def alignement(self, boids, params):

        """"Calcule la tendance à aligner la vitesse avec celle des voisins """

        avg_vel = np.zeros(2)  # moyenne des vitesses des voisins
        total_voisins=0 #compteurs de voisins
        r_align = params['r_alignement']  # rayon de perception défini dans les paramètres (zone d'influence des voisins"

        for other in boids:
            if other is self:   # On ne s'aligne pas sur soi-même
                continue
            distance= np.linalg.norm(self.pos-other.pos) #Distance entre soi et l'autre boid
            if distance<r_align:  #Si l'autre boid est dans le rayon de perception
                avg_vel+=other.vel #On ajoute sa vitesse à la moyenne
                total_voisins+=1 #on incrémente le nombre de voisins

        if total_voisins>0:  #si on a des voisins proches
            avg_vel/=total_voisins # on recalcule la vitesse moyenne
            steer=avg_vel-self.vel # on ajuste la vitesse vers cette moyenne
            return steer
        else:
            return np.zeros(2)  # aucun voisin donc pas de changement


    # ---------------------------------------------------------------------
    # Règle de cohésion : aller le centre du groupe

    def cohesion(self, boids, params):

        """Calcule une force d'attraction vers le centre de masse des voisins"""

        centre_mass=np.zeros(2)  #centre moyen des positions voisines
        total_voisins=0       #on initialise le nombre de voisins
        r_coh=params['r_cohesion']  #rayon de perception pour la cohésion

        for other in boids:
            if other is self:   #on ignore soi-même
                continue
            distance=np.linalg.norm(self.pos-other.pos)
            if distance<r_coh: #si le voisin est assez proche du rayon de centre de masse
                centre_mass+=other.pos # on ajoute sa position
                total_voisins+=1

        if total_voisins>0:
            centre_mass/=total_voisins  #centre du groupe (moyenne des positions)
            steer=centre_mass-self.pos #direction vers le centre
            return steer
        else:
            return np.zeros(2) # pas de changement


    # ---------------------------------------------------------------------
    # Limitation de la vitesse

    def limit_speed(self, max_speed):
        """ Limite la norme de la vitesse d'un boid à une
        valeur maximale pour éviter les accélérations trop fortes"""

        speed = np.linalg.norm(self.vel) #on calcule la vitesse actuelle
        if speed > max_speed:             # si elle dépasse la vitesse maximale autorisée
            self.vel = (self.vel / speed) * max_speed  # alors, on la réduit considérablement


    # ---------------------------------------------------------------------
    # Gestion des frontières
    def apply_boundaries(self, width, height):
        """"
        Si le boid sort du cadre (par exemple 100x100)
        on le fait réapparaître de l'autre côté — effet tore"""
        if self.pos[0]<0: #si le boid dépasse à gauche
            self.pos[0]=width  #il réapparaît à droite
        elif self.pos[0]>width: # s'il dépasse à droite
            self.pos[0]=0  #il revient à gauche

        if self.pos[1]<0:  #si le boid dépasse en bas
            self.pos[1]=height  #il revient en haut
        elif self.pos[1]>height: # s'il dépasse en haut
            self.pos[1]=0       #il revient en bas


    # Méthode de mise à jour de la position et de la vitesse
    # ---------------------------------------------------------------------
    def update(self,boids,params):
        """ Mets à jour la vitesse et la position du Boid à chaque pas de temps selon le model Boids
        param boids : liste de tous les boids présents dans la simulation
        param params : dictionnaire contenant les paramètres du modèle
        """

        #Calcule de chaque force comportementale
        separation=self.separation(boids,params) #force de séparation
        alignement = self.alignement(boids, params)  # force de séparation
        cohesion = self.cohesion(boids, params)  # force de séparation

        # Combinaison pondérée des trois forces selon les poids définis
        self.vel+= (params['w_separation'] * separation +
                    params['w_alignement'] * alignement +
                    params['w_cohesion'] * cohesion)

        #Limiter la vitesse pour éviter les explosions numériques
        self.limit_speed(params['max_speed'])

        #Mise à jour de la position
        self.pos+= self.vel

