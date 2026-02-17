import unittest
import numpy as np
from KNNAlgorithm.CalculateDistance.Strategy.EuclideanDistance import EuclideanDistance
from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm


class TestProgettoAI(unittest.TestCase):

    # TEST 1: Verifica il calcolo della Distanza Euclidea
    def test_euclidean_distance(self):
        strategy = EuclideanDistance()
        p1 = np.array([1, 1])
        p2 = np.array([4, 5])
        # Distanza euclidea quadrata (come implementato nel tuo codice):
        # (1-4)^2 + (1-5)^2 = 9 + 16 = 25
        dist = strategy.calculate(p1, p2)
        self.assertEqual(dist, 25.0, "Il calcolo della distanza euclidea è errato")

    # TEST 2: Verifica il funzionamento base del KNN (piccolo dataset dummy)
    def test_knn_prediction(self):
        X_train = [[1, 1], [1, 2], [5, 5], [5, 6]]
        y_train = [0, 0, 1, 1]

        knn = KnnAlgorithm(k=1)
        knn.fit(X_train, y_train)

        # Punto vicino a (1,1) -> deve essere 0
        pred = knn.predict([[1.5, 1.5]])
        self.assertEqual(pred[0], 0, "Il KNN ha fallito la classificazione del gruppo 0")

    # TEST 3: Verifica eccezione per k non valido
    def test_knn_invalid_k(self):
        # Deve alzare un ValueError se k <= 0
        with self.assertRaises(ValueError):
            KnnAlgorithm(k=0)


if __name__ == '__main__':
    unittest.main()