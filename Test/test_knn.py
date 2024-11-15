import numpy as np
from mon_knn.knn import KNN

def test_knn():
    # Données d'entraînement
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([0, 0, 1, 1, 1])
    
    # Créer le classificateur KNN avec k=3
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    
    # Données de test
    X_test = np.array([[2, 2], [4, 4]])
    predictions = knn.predict(X_test)
    
    assert len(predictions) == 2
    print("Test réussi :", predictions)

if __name__ == "__main__":
    test_knn()