import unittest

import mock


class Test_get_symbolic_vandermonde(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(p_order, x_vals=None):
        from assignment1.dg1_symbolic import get_symbolic_vandermonde
        return get_symbolic_vandermonde(p_order, x_vals=x_vals)

    def test_p1(self):
        import sympy

        x_symb = sympy.Symbol('x')
        x_vec, vand_mat = self._call_func_under_test(1)
        self.assertTrue(isinstance(x_vec, sympy.Matrix))
        self.assertEqual(x_vec.tolist(), [[1, x_symb]])
        self.assertTrue(isinstance(vand_mat, sympy.Matrix))
        expected_vand = [
            [1, 0],
            [1, 1],
        ]
        self.assertEqual(vand_mat.tolist(), expected_vand)

    def test_p4(self):
        import sympy

        x_symb = sympy.Symbol('x')
        x_vec, vand_mat = self._call_func_under_test(4)
        self.assertTrue(isinstance(x_vec, sympy.Matrix))
        self.assertEqual(x_vec.tolist(),
                         [[1, x_symb, x_symb**2, x_symb**3, x_symb**4]])
        self.assertTrue(isinstance(vand_mat, sympy.Matrix))
        expected_vand = [
            [256, 0, 0, 0, 0],
            [256, 64, 16, 4, 1],
            [256, 128, 64, 32, 16],
            [256, 192, 144, 108, 81],
            [256, 256, 256, 256, 256],
        ]
        self.assertEqual((256 * vand_mat).tolist(), expected_vand)


class Test_find_matrices_symbolic(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(p_order, start=0, stop=1, x_vals=None):
        from assignment1.dg1_symbolic import find_matrices_symbolic
        return find_matrices_symbolic(p_order, start=start,
                                      stop=stop, x_vals=x_vals)

    def test_p1(self):
        import sympy

        mass_mat, stiffness_mat = self._call_func_under_test(1)
        self.assertTrue(isinstance(mass_mat, sympy.Matrix))
        self.assertEqual((6 * mass_mat).tolist(),
                         [[2, 1], [1, 2]])
        self.assertTrue(isinstance(stiffness_mat, sympy.Matrix))
        self.assertEqual((2 * stiffness_mat).tolist(),
                         [[-1, -1], [1, 1]])

    def test_p2(self):
        import sympy

        mass_mat, stiffness_mat = self._call_func_under_test(2)
        self.assertTrue(isinstance(mass_mat, sympy.Matrix))
        self.assertEqual((30 * mass_mat).tolist(),
                         [[4, 2, -1], [2, 16, 2], [-1, 2, 4]])
        self.assertTrue(isinstance(stiffness_mat, sympy.Matrix))
        self.assertEqual((6 * stiffness_mat).tolist(),
                         [[-3, -4, 1], [4, 0, -4], [-1, 4, 3]])

    def test_p3(self):
        import sympy

        mass_mat, stiffness_mat = self._call_func_under_test(3)
        self.assertTrue(isinstance(mass_mat, sympy.Matrix))
        self.assertEqual((1680 * mass_mat).tolist(), [
            [128, 99, -36, 19],
            [99, 648, -81, -36],
            [-36, -81, 648, 99],
            [19, -36, 99, 128],
        ])
        self.assertTrue(isinstance(stiffness_mat, sympy.Matrix))
        self.assertEqual((80 * stiffness_mat).tolist(), [
            [-40, -57, 24, -7],
            [57, 0, -81, 24],
            [-24, 81, 0, -57],
            [7, -24, 57, 40],
        ])

    def test_p3_gauss_lobatto(self):
        import sympy

        sq5 = sympy.sqrt(5)
        x_vals = [-1, -1/sq5, 1/sq5, 1]
        mass_mat, stiffness_mat = self._call_func_under_test(
            3, start=-1, stop=1, x_vals=x_vals)
        self.assertTrue(isinstance(mass_mat, sympy.Matrix))
        self.assertEqual((42 * mass_mat).tolist(), [
            [6, sq5, -sq5, 1],
            [sq5, 30, 5, -sq5],
            [-sq5, 5, 30, sq5],
            [1, -sq5, sq5, 6],
        ])
        self.assertTrue(isinstance(stiffness_mat, sympy.Matrix))
        P1 = sympy.Matrix([
            [-12, -5, -5, -2],
            [5, 0, 0, -5],
            [5, 0, 0, -5],
            [2, 5, 5, 12],
        ])
        P2 = sympy.Matrix([
            [0, -5, 5, 0],
            [5, 0, -10, 5],
            [-5, 10, 0, -5],
            [0, -5, 5, 0],
        ])
        self.assertEqual(24 * stiffness_mat, P1 + sq5 * P2)
