import unittest

import mock


class Test_make_lagrange_matrix(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(x_vals, all_x):
        from assignment1.plotting import make_lagrange_matrix
        return make_lagrange_matrix(x_vals, all_x)

    def test_it(self):
        import numpy as np

        x_vals = np.array([0., 1., 2.])
        # The Lagrange functions for the interpolating x values 0, 1, 2
        # are simply L0(x) = (x - 1)(x - 2)/2, L1(x) = x(2 - x) and
        # L2(x) = x(x - 1)/2.
        all_x = np.array([3., 5., 8., 13.])
        result = self._call_func_under_test(x_vals, all_x)
        expected_result = [
            [1, -3, 3],
            [6, -15, 10],
            [21, -48, 28],
            [66, -143, 78],
        ]
        self.assertTrue(np.all(result == expected_result))


class TestPolynomialInterpolate(unittest.TestCase):

    @staticmethod
    def _get_target_class():
        from assignment1.plotting import PolynomialInterpolate
        return PolynomialInterpolate

    def _make_one(self, x_vals, num_points=None):
        return self._get_target_class()(x_vals, num_points=num_points)

    def test_constructor_defaults(self):
        import numpy as np

        x_vals = np.array([0.0, 1.0])
        num_points = 3
        with mock.patch('assignment1.plotting.INTERVAL_POINTS',
                        new=num_points):
            interp_func = self._make_one(x_vals)

        self.assertTrue(interp_func.x_vals is x_vals)
        # Split the interval [0, 1] into ``num_points`` == 3 points.
        self.assertTrue(np.all(interp_func.all_x == [0, 0.5, 1.0]))
        # The Lagrange functions for the interpolating x values 0, 1
        # are simply L0(x) = 1 - x and L1(x) = x. Evaluating these
        # at [0, 0.5, 1] is straightforward.
        expected_lagrange = [
            [1, 0],
            [0.5, 0.5],
            [0, 1],
        ]
        self.assertTrue(
            np.all(interp_func.lagrange_matrix == expected_lagrange))

    def test_constructor_explicit(self):
        import numpy as np

        x_vals = np.array([0.0, 1.0, 2.0])
        num_points = 5
        interp_func = self._make_one(x_vals, num_points=num_points)

        self.assertTrue(interp_func.x_vals is x_vals)
        # Split the interval [0, 2] into ``num_points`` == 5 points.
        self.assertTrue(np.all(interp_func.all_x == [0, 0.5, 1.0, 1.5, 2.0]))
        # The Lagrange functions for the interpolating x values 0, 1, 2
        # are simply L0(x) = (x - 1)(x - 2)/2, L1(x) = x(2 - x) and
        # L2(x) = x(x - 1)/2. Evaluating these at [0, 0.5, 1, 1.5, 2] gives.
        expected_lagrange = [
            [8, 0, 0],
            [3, 6, -1],
            [0, 8, 0],
            [-1, 6, 3],
            [0, 0, 8],
        ]
        self.assertTrue(
            np.all(8 * interp_func.lagrange_matrix == expected_lagrange))

    def test_from_solver_factory(self):
        import numpy as np

        klass = self._get_target_class()
        solver = mock.MagicMock(node_points=np.array([
            [0., 2., 4.],
            [1., 3., 5.],
        ]))
        num_points = 5
        interp_func = klass.from_solver(solver, num_points=num_points)
        # x_vals are first column of mock node points
        self.assertTrue(np.all(interp_func.x_vals == [0, 1]))
        # Split the interval [0, 1] (given by the first column of x_vals)
        # into num_points = 5 pieces.
        self.assertTrue(np.all(interp_func.all_x ==
                               [0.0, 0.25, 0.5, 0.75, 1.0]))
        # The interpolating functions are L0(x) = 1 - x, L1(x) = x, evaluated
        # at the points above.
        expected_lagrange = [
            [1.0, 0.0],
            [0.75, 0.25],
            [0.5, 0.5],
            [0.25, 0.75],
            [0.0, 1.0],
        ]
        self.assertTrue(np.all(interp_func.lagrange_matrix ==
                               expected_lagrange))

    def test_interpolate_1D_data(self):
        import numpy as np

        x_vals = np.array([0.0, 1.0, 2.0])
        interp_func = self._make_one(x_vals, num_points=5)
        y_vals = np.array([2., 3., 2.])
        # Our polynomial is
        #        2 L0(x) + 3 L1(x) + 2 L2(x)
        #     == (x - 1)(x - 2) + 3x(2 - x) + x(x - 1)
        #     == - x^2 + 2x + 2
        # and we evaluate it at the 5 points [0, 0.5, 1.0, 1.5, 2.0].
        self.assertEqual(y_vals.shape, (3,))
        result = interp_func.interpolate(y_vals)
        # Verify it becomes 2D.
        self.assertEqual(result.shape, (5, 1))
        expected_result = [
            [2.],
            [2.75],
            [3.],
            [2.75],
            [2.],
        ]
        self.assertTrue(np.all(result == expected_result))

    def test_interpolate_2D_data(self):
        import numpy as np

        x_vals = np.array([0.0, 1.0])
        interp_func = self._make_one(x_vals, num_points=3)
        y_vals = np.array([
            [10., 11., 5.],
            [20., 12., 4.],
        ])
        # Our polynomials are
        #     10 L0(x) + 20 L1(x) = 10(1 - x) + 20x = 10 + 10x
        #     11 L0(x) + 12 L1(x) = 11(1 - x) + 12x = 11 +   x
        #      5 L0(x) +  4 L1(x) =  5(1 - x) +  4x =  5 -   x
        # and we evaluate them at the 3 points [0, 0.5, 2.0]
        self.assertEqual(y_vals.shape, (2, 3))
        result = interp_func.interpolate(y_vals)
        # Verify it becomes 2D.
        self.assertEqual(result.shape, (3, 3))
        expected_result = [
            [10, 11, 5],
            [15, 11.5, 4.5],
            [20, 12, 4],
        ]
        self.assertTrue(np.all(result == expected_result))


class Test_plot_solution(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(color, num_cols, interp_func,
                              solver, ax):
        from assignment1.plotting import plot_solution
        return plot_solution(color, num_cols, interp_func, solver, ax)

    def test_it(self):
        import sympy

        # Create mock DG1Solver.
        node_points = sympy.Matrix([
            [42, 1337],
        ])
        solution = object()
        solver = mock.MagicMock(
            node_points=node_points,
            solution=solution,
        )
        # Create mock PolynomialInterpolate.
        all_x = 99
        interp_func = mock.MagicMock(
            all_x=all_x,
        )
        y_vals = sympy.Matrix([
            [10, 100],
            [11, 200],
        ])
        interp_func.interpolate.return_value = y_vals
        # Create mock axis.
        ax = mock.MagicMock()
        line_obj = object()
        ax.plot.return_value = (line_obj,)
        # Call the method.
        num_cols = 2
        color = 'green'
        result = self._call_func_under_test(
            color, num_cols, interp_func, solver, ax)
        # Verify result (lines come from ax.plot mock).
        self.assertEqual(result, [line_obj] * num_cols)
        # Verify mocks were called.
        interp_func.interpolate.assert_called_once_with(solution)
        plot_calls = ax.plot.mock_calls
        self.assertEqual(len(plot_calls), 2)
        self.assertEqual(
            plot_calls[0],
            mock.call(
                all_x + node_points[0, 0],
                y_vals[:, 0],
                color=color, linewidth=2,
            ),
        )
        self.assertEqual(
            plot_calls[1],
            mock.call(
                all_x + node_points[0, 1],
                y_vals[:, 1],
                color=color, linewidth=2,
            ),
        )


class Test__configure_axis(unittest.TestCase):

    @staticmethod
    def _call_func_under_test(ax, **kwargs):
        from assignment1.plotting import _configure_axis
        return _configure_axis(ax, **kwargs)

    def test_defaults(self):
        # Make ax a mock.
        ax = mock.MagicMock()
        # Call function.
        result = self._call_func_under_test(ax)
        self.assertEqual(result, None)
        # Verify mocks were called.
        ax.set_xlim.assert_called_once_with(0.0, 1.0)
        ax.set_ylim.assert_called_once_with(-0.1, 1.1)
        ax.grid.assert_called_once_with(b=True)

    def test_explicit(self):
        # Make ax a mock.
        ax = mock.MagicMock()
        # Call function.
        x_min = 1.2
        x_max = 3.4
        y_min = 5.6
        y_max = 7.8
        y_buffer = 1000.0
        result = self._call_func_under_test(ax, x_min=x_min, x_max=x_max,
                                            y_min=y_min, y_max=y_max,
                                            y_buffer=y_buffer)
        self.assertEqual(result, None)
        # Verify mocks were called.
        ax.set_xlim.assert_called_once_with(x_min, x_max)
        ax.set_ylim.assert_called_once_with(y_min - y_buffer,
                                            y_max + y_buffer)
        ax.grid.assert_called_once_with(b=True)


class TestDG1Animate(unittest.TestCase):

    @staticmethod
    def _get_target_class():
        from assignment1.plotting import DG1Animate
        return DG1Animate

    def _make_one(self, solver, fig=None, ax=None, interp_points=None):
        return self._get_target_class()(solver, fig=fig, ax=ax,
                                        interp_points=interp_points)

    @mock.patch('matplotlib.pyplot.subplots', create=True)
    @mock.patch('assignment1.plotting.PolynomialInterpolate.from_solver',
                return_value=object())
    def test_constructor_defaults(self, interp_factory, subplots):
        # Set-up mocks.
        fig = object()
        ax = object()
        subplots.return_value = fig, ax
        # Construct the object.
        solver = object()
        animate_obj = self._make_one(solver)
        # Verify properties
        self.assertEqual(animate_obj.solver, solver)
        self.assertEqual(animate_obj.poly_interp_func,
                         interp_factory.return_value)
        self.assertEqual(animate_obj.fig, fig)
        self.assertEqual(animate_obj.ax, ax)
        # Verify mock.
        subplots.assert_called_once_with(1, 1)
        interp_factory.assert_called_once_with(solver, num_points=None)

    @mock.patch('assignment1.plotting.PolynomialInterpolate.from_solver')
    def test_constructor_axis_with_no_figure(self, interp_factory):
        # Make a subclass which mocks out the constructor helpers.
        solver = object()
        # Fail to construct the object.
        with self.assertRaises(ValueError):
            self._make_one(solver, ax=object())
        interp_factory.assert_called_once_with(solver, num_points=None)

    @mock.patch('assignment1.plotting.PolynomialInterpolate.from_solver')
    def test_constructor_figure_with_no_axis(self, interp_factory):
        # Make a subclass which mocks out the constructor helpers.
        solver = object()
        # Fail to construct the object.
        with self.assertRaises(ValueError):
            self._make_one(solver, fig=object())
        interp_factory.assert_called_once_with(solver, num_points=None)

    @mock.patch('assignment1.plotting.PolynomialInterpolate.from_solver',
                return_value=object())
    @mock.patch('assignment1.plotting._configure_axis')
    @mock.patch('assignment1.plotting.plot_solution')
    def test_init_func(self, plot_soln, config_ax, interp_factory):
        import numpy as np

        # Set-up mocks.
        num_cols = 1
        solver = mock.MagicMock(
            node_points=np.array([0, 1]).reshape(2, num_cols),
            solution=[2, 3],
        )
        fig = object()
        ax = object()
        plot_line = mock.MagicMock()
        plot_soln.return_value = [plot_line]
        # Construct the object and call the method under test.
        animate_obj = self._make_one(solver, fig=fig, ax=ax)
        plot_lines = animate_obj.init_func()
        self.assertEqual(plot_lines, plot_soln.return_value)
        # Verify mocks were called.
        config_ax.assert_called_once_with(ax, x_min=0, x_max=1,
                                          y_min=2, y_max=3)
        self.assertEqual(len(plot_soln.mock_calls), 2)
        self.assertEqual(
            plot_soln.mock_calls[0],
            mock.call('red', num_cols, interp_factory.return_value,
                      solver, ax),
        )
        interp_factory.assert_called_once_with(solver, num_points=None)
        self.assertEqual(
            plot_soln.mock_calls[1],
            mock.call('blue', num_cols, interp_factory.return_value,
                      solver, ax),
        )
        interp_factory.assert_called_once_with(solver, num_points=None)
        plot_line.set_visible.assert_called_once_with(False)

    @mock.patch('assignment1.plotting.PolynomialInterpolate.from_solver')
    def test_update_plot_frame_zero(self, interp_factory):
        import sympy

        # Set-up mocks.
        solver = mock.MagicMock(
            current_step=0,
            solution=object(),
        )
        fig = ax = object()
        poly_interp = mock.MagicMock()
        poly_interp.interpolate.return_value = sympy.Matrix([[0, 0]])
        interp_factory.return_value = poly_interp
        plot_line = mock.MagicMock()
        # Construct the object and call the method under test.
        animate_obj = self._make_one(solver, fig=fig, ax=ax)
        animate_obj.plot_lines = [plot_line]
        plot_lines = animate_obj.update_plot(solver.current_step)
        self.assertEqual(plot_lines, animate_obj.plot_lines)
        # Verify mocks were called.
        plot_line.set_visible.assert_called_once_with(True)
        solver.update.assert_called_once_with()
        poly_interp.interpolate.assert_called_once_with(solver.solution)
        plot_line.set_ydata.assert_called_once_with(sympy.Matrix([[0]]))

    @mock.patch('assignment1.plotting.PolynomialInterpolate.from_solver')
    def test_update_plot_frame_nonzero(self, interp_factory):
        import sympy

        # Set-up mocks.
        solver = mock.MagicMock(
            current_step=10203,
            solution=object(),
        )
        fig = ax = object()
        poly_interp = mock.MagicMock()
        poly_interp.interpolate.return_value = sympy.Matrix([[0, 0]])
        interp_factory.return_value = poly_interp
        plot_line = mock.MagicMock()
        # Construct the object and call the method under test.
        animate_obj = self._make_one(solver, fig=fig, ax=ax)
        animate_obj.plot_lines = [plot_line]
        plot_lines = animate_obj.update_plot(solver.current_step)
        self.assertEqual(plot_lines, animate_obj.plot_lines)
        # Verify mocks were called.
        plot_line.set_visible.assert_not_called()
        solver.update.assert_called_once_with()
        poly_interp.interpolate.assert_called_once_with(solver.solution)
        plot_line.set_ydata.assert_called_once_with(sympy.Matrix([[0]]))

    @mock.patch('assignment1.plotting.PolynomialInterpolate.from_solver')
    def test_update_plot_mismatch_frame(self, interp_factory):
        # Set-up mocks.
        solver = mock.MagicMock(
            current_step=10203,
            solution=object(),
        )
        fig = ax = object()
        poly_interp = mock.MagicMock()
        interp_factory.return_value = poly_interp
        plot_line = mock.MagicMock()
        # Construct the object and call the method under test.
        animate_obj = self._make_one(solver, fig=fig, ax=ax)
        animate_obj.plot_lines = [plot_line]
        with self.assertRaises(ValueError):
            animate_obj.update_plot(-1)
        # Verify mocks were called.
        plot_line.set_visible.assert_not_called()
        solver.update.assert_not_called()
        poly_interp.interpolate.assert_not_called()
        plot_line.set_ydata.assert_not_called()
