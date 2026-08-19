import unittest

import numpy as np

from csub_snp_virulence.distance import pairwise_ibs_distance, virulence_hamming_distance


class CoreDistanceTests(unittest.TestCase):
    def test_pairwise_ibs_distance(self):
        dosage = np.array([[0, 1, 2], [0, 2, 0], [2, 1, np.nan]], dtype=float)
        distance, overlap = pairwise_ibs_distance(dosage)
        self.assertAlmostEqual(distance[0, 1], (0 + 0.5 + 1.0) / 3)
        self.assertAlmostEqual(distance[0, 2], (1.0 + 0.0) / 2)
        self.assertEqual(overlap[0, 2], 2)

    def test_virulence_hamming_distance(self):
        response = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.int8)
        distance = virulence_hamming_distance(response)
        self.assertAlmostEqual(distance[0, 1], 2 / 3)
        self.assertEqual(distance[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
