import unittest


class Test_find_matrices_symbolic(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(p_order):
        from assignment1.dg1 import find_matrices_symbolic
        return find_matrices_symbolic(p_order)
