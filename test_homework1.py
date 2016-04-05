"""Example unit tests for Homework #1

Important
=========

Do not modify the way in which your solution functions

* homework1.exercise1.newton_step
* homework1.exercise1.newton

are imported. The actual test suite used to grade your homework will import
your functions in the exact same way.

"""

import unittest
import numpy
from numpy import sin, cos, exp, pi, dot, eye, zeros, ones, array
from numpy.linalg import norm
from numpy.random import randn

# Import the homework functions
from homework1.exercise1 import collatz_step, collatz
from homework1.exercise2 import gradient_step, gradient_descent
from homework1.exercise3 import (
    decompose,
    jacobi_step,
    jacobi_iteration,
    gauss_seidel_step,
    gauss_seidel_iteration,
)

class TestExercise1(unittest.TestCase):
    """Testing the validity of

    * homework1.exercise1.collatz_step
    * homework1.exercise1.collatz
    """
    def test_collatz_step(self):
        self.assertEqual(collatz_step(5), 16)
        self.assertEqual(collatz_step(16), 8)

    def test_collatz_step_one(self):
        self.assertEqual(collatz_step(1), 1)

    def test_collatz_step_error(self):
        with self.assertRaises(ValueError):
            collatz_step(-1)
            collatz_step(-2)

    def test_collatz(self):
        s6 = [6, 3, 10, 5, 16, 8, 4, 2, 1]
        self.assertEqual(collatz(6), s6)

        s43 = [43, 130, 65, 196, 98, 49, 148, 74, 37, 112, 56, 28, 14, 7, 22,
               11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
        self.assertEqual(collatz(43), s43)


class TestExercise2(unittest.TestCase):
    """Testing the validity of

    * homework1.exercise2.gradient_step
    * homework1.exercise2.gradient_descent

    """
    def test_gradient_step(self):
        f = lambda x: x**2 - 1
        df = lambda x: 2*x
        x0 = 1
        x1 = gradient_step(x0, df, sigma=0.25)
        x1_actual = 0.5 # x0 - sigma*(2*x0)
        self.assertAlmostEqual(x1, x1_actual)

    def test_sigma_condition(self):
        f = lambda x: x**2 - 1
        df = lambda x: 2*x
        x0 = 1
        with self.assertRaises(ValueError):
            gradient_descent(f, df, x0, sigma=-100)
            gradient_descent(f, df, x0, sigma=100)


class TestExercise3(unittest.TestCase):
    """Testing the validity of

    * homework1.exercise3.
    * homework1.exercise3.
    * homework1.exercise3.
    * homework1.exercise3.
    * homework1.exercise3.
    * homework1.exercise3.

    """
    def test_decompose(self):
        # the test written below only tests if the identity matrix is properly
        # decomposed. this is not sufficient for testing if decompose() works
        # properly but is a good start.
        A = eye(3)
        D, L, U = decompose(A)
        D_actual = eye(3)
        L_actual = zeros((3,3))
        U_actual = zeros((3,3))

        self.assertAlmostEqual(norm(D_actual - D), 0)
        self.assertAlmostEqual(norm(L_actual - L), 0)
        self.assertAlmostEqual(norm(U_actual - U), 0)

    def test_jacobi_step(self):
        # the test written below only tests if jacobi step works in the case
        # when A is the identity matrix. In this case, jacobi_step() should
        # converge immediately the the answer. (Can you see why based on the
        # definition of Jacobi iteration?) This is not sufficient for testing
        # if jacobi_step() works properly but is a good start.
        D = eye(3)
        L = zeros((3,3))
        U = zeros((3,3))
        b = array([1,2,3])
        x0 = ones(3)
        x1 = jacobi_step(D, L, U, b, x0)

        self.assertAlmostEqual(norm(x1-b), 0)

    def test_jacobi_iteration(self):
        # the test written below only tests if jacobi iteration works in the
        # case when A is the identity matrix.
        A = eye(3)
        b = array([1,2,3])
        x0 = ones(3)
        x = jacobi_iteration(A, b, x0)

        self.assertAlmostEqual(norm(x-b), 0)



# The following code is run when this Python module / file is executed as a
# script. This happens when you enter
#
# $ python test_homework1.py
#
# in the terminal.
if __name__ == '__main__':
    unittest.main(verbosity=2) # run the above tests
