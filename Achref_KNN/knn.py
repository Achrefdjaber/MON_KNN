import numpy as np

class KNN:
    def __init__(self, k=3):
        """
        Initialiser le classificateur KNN.
        k: nombre de voisins à considérer (par défaut = 3)
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Entraîner le classificateur KNN.
        X: matrice des caractéristiques (n_samples, n_features)
        y: vecteur des étiquettes (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        """
        Calculer la distance euclidienne entre deux points.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """
        Prédire les classes pour chaque échantillon dans X.
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        """
        Prédire la classe pour un échantillon unique x.
        """
        # Calculer la distance entre x et tous les échantillons d'entraînement
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Trouver les indices des k plus proches voisins
        k_indices = np.argsort(distances)[:self.k]
        
        # Extraire les étiquettes des k plus proches voisins
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Retourner la classe majoritaire
        return np.bincount(k_nearest_labels).argmax()
