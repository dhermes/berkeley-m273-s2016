import unittest


class Test_get_symbolic_vandermonde(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(p_order):
        from assignment1.dg1 import get_symbolic_vandermonde
        return get_symbolic_vandermonde(p_order)

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
    def _call_func_under_test(p_order):
        from assignment1.dg1 import find_matrices_symbolic
        return find_matrices_symbolic(p_order)

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


class Test_mass_and_stiffness_matrices_p1(unittest.TestCase):

    @staticmethod
    def _call_func_under_test():
        from assignment1.dg1 import mass_and_stiffness_matrices_p1
        return mass_and_stiffness_matrices_p1()

    def test_value(self):
        import numpy as np

        # Make sure multiple calls give identical data.
        mass_mat, stiffness_mat = self._call_func_under_test()
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))

        self.assertTrue(np.all(6 * mass_mat == [[2, 1], [1, 2]]))
        self.assertTrue(np.all(2 * stiffness_mat == [[-1, -1], [1, 1]]))

    def test_identical(self):
        import numpy as np

        # Make sure multiple calls give identical data.
        mass_mat1, stiffness_mat1 = self._call_func_under_test()
        self.assertTrue(isinstance(mass_mat1, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat1, np.ndarray))
        mass_mat2, stiffness_mat2 = self._call_func_under_test()

        self.assertFalse(mass_mat1 is mass_mat2)
        self.assertTrue(np.all(mass_mat1 == mass_mat2))
        self.assertFalse(stiffness_mat1 is stiffness_mat2)
        self.assertTrue(np.all(stiffness_mat1 == stiffness_mat2))


class Test_mass_and_stiffness_matrices_p2(unittest.TestCase):

    @staticmethod
    def _call_func_under_test():
        from assignment1.dg1 import mass_and_stiffness_matrices_p2
        return mass_and_stiffness_matrices_p2()

    def test_value(self):
        import numpy as np

        # Make sure multiple calls give identical data.
        mass_mat, stiffness_mat = self._call_func_under_test()
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))

        expected_mass_mat = [[4, 2, -1], [2, 16, 2], [-1, 2, 4]]
        self.assertTrue(np.all(30 * mass_mat == expected_mass_mat))
        expected_stiffness_mat = [[-3, -4, 1], [4, 0, -4], [-1, 4, 3]]
        self.assertTrue(np.all(6 * stiffness_mat == expected_stiffness_mat))

    def test_identical(self):
        import numpy as np

        # Make sure multiple calls give identical data.
        mass_mat1, stiffness_mat1 = self._call_func_under_test()
        self.assertTrue(isinstance(mass_mat1, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat1, np.ndarray))
        mass_mat2, stiffness_mat2 = self._call_func_under_test()

        self.assertFalse(mass_mat1 is mass_mat2)
        self.assertTrue(np.all(mass_mat1 == mass_mat2))
        self.assertFalse(stiffness_mat1 is stiffness_mat2)
        self.assertTrue(np.all(stiffness_mat1 == stiffness_mat2))


class Test_mass_and_stiffness_matrices_p3(unittest.TestCase):

    @staticmethod
    def _call_func_under_test():
        from assignment1.dg1 import mass_and_stiffness_matrices_p3
        return mass_and_stiffness_matrices_p3()

    def test_value(self):
        import numpy as np

        # Make sure multiple calls give identical data.
        mass_mat, stiffness_mat = self._call_func_under_test()
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))

        expected_mass_mat = [
            [128, 99, -36, 19],
            [99, 648, -81, -36],
            [-36, -81, 648, 99],
            [19, -36, 99, 128],
        ]
        self.assertTrue(np.all(1680 * mass_mat == expected_mass_mat))
        expected_stiffness_mat = [
            [-40, -57, 24, -7],
            [57, 0, -81, 24],
            [-24, 81, 0, -57],
            [7, -24, 57, 40],
        ]
        self.assertTrue(np.all(80 * stiffness_mat == expected_stiffness_mat))

    def test_identical(self):
        import numpy as np

        # Make sure multiple calls give identical data.
        mass_mat1, stiffness_mat1 = self._call_func_under_test()
        self.assertTrue(isinstance(mass_mat1, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat1, np.ndarray))
        mass_mat2, stiffness_mat2 = self._call_func_under_test()

        self.assertFalse(mass_mat1 is mass_mat2)
        self.assertTrue(np.all(mass_mat1 == mass_mat2))
        self.assertFalse(stiffness_mat1 is stiffness_mat2)
        self.assertTrue(np.all(stiffness_mat1 == stiffness_mat2))


class Test_find_matrices(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(p_order):
        from assignment1.dg1 import find_matrices
        return find_matrices(p_order)

    def test_p1(self):
        import numpy as np
        from assignment1.dg1 import mass_and_stiffness_matrices_p1

        mass_mat, stiffness_mat = self._call_func_under_test(1)
        (expected_mass_mat,
         expected_stiffness_mat) = mass_and_stiffness_matrices_p1()
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))
        self.assertTrue(np.allclose(mass_mat, expected_mass_mat))
        self.assertTrue(np.allclose(stiffness_mat, expected_stiffness_mat))

    def test_p2(self):
        import numpy as np
        from assignment1.dg1 import mass_and_stiffness_matrices_p2

        mass_mat, stiffness_mat = self._call_func_under_test(2)
        (expected_mass_mat,
         expected_stiffness_mat) = mass_and_stiffness_matrices_p2()
        self.assertTrue(isinstance(mass_mat, np.ndarray))
        self.assertTrue(isinstance(stiffness_mat, np.ndarray))
        self.assertTrue(np.allclose(mass_mat, expected_mass_mat))
        self.assertTrue(np.allclose(stiffness_mat, expected_stiffness_mat))
