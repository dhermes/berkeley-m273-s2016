import unittest


class Test_find_matrices_symbolic(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(p_order):
        from assignment1.dg1 import find_matrices_symbolic
        return find_matrices_symbolic(p_order)

    def test_p1(self):
        mass_mat, stiffness_mat = self._call_func_under_test(1)
        self.assertEqual((6 * mass_mat).tolist(),
                         [[2, 1], [1, 2]])
        self.assertEqual((2 * stiffness_mat).tolist(),
                         [[-1, -1], [1, 1]])

    def test_p2(self):
        mass_mat, stiffness_mat = self._call_func_under_test(2)
        self.assertEqual((30 * mass_mat).tolist(),
                         [[4, 2, -1], [2, 16, 2], [-1, 2, 4]])
        self.assertEqual((6 * stiffness_mat).tolist(),
                         [[-3, -4, 1], [4, 0, -4], [-1, 4, 3]])
