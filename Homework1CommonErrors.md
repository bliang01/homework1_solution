# General issues

A few issues cropped up repeatedly. One is that code copied-and-pasted from a
Jupyter notebook often doesn't work right off the bat (for instance, it may
refer to other code that you forgot to copy and paste). It's important when
moving code around to test that the result works. In this case, since the code
needed to end up in `.py` files anyway, it would have been easier to write and
test the code there first, and load that code into a Jupyter notebook with an
import statement.

Another common mistake was to write a loop looking like this for
`gradient_descent` or one of the Exercise 3 functions:

```python
 while (norm(jacobi_step(D, L, U, b, xk) - xk) > epsilon):
     xk = jacobi_step(D, L, U, b, xk)
 return xk
```

This kind of loop has three problems:

1. Every time the loop executes, `jacobi_step` is called twice (once in the
   `while` statement, and once to assign `xk`). This is wasteful, since you can
   just store the value.

2. The last time the loop executes, `jacobi_step` is called on `xk`, but then
   that value is never stored in `xk`. So effectively, the very last iteration
   is thrown away, and the second-last iteration is returned instead. Why not
   go ahead and use the very best value you've calculated?

3. Making long function calls in a condition makes it harder to understand the
   flow of code.

It's generally better to make a simple change like this:

```python
 # No harm in doing one step immediately.
 xkp1 = jacobi_step(D, L, U, b, xk)
 # Now loop until the next iteration is less than epsilon from xk.
 while (norm(xkp1 - xk) > epsilon):
     xk = xkp1
     xkp1 = jacobi_step(D, L, U, b, xk)
 return xkp1
```

By storing the next value of `xk` in `xkp1`, we get rid of the extra call to
`jacobi_step` in each iteration, and make it possible to return this value.

One last common mistake was to use different convergence criteria from what the
assignment requested (e.g. using the 1-norm or max-norm rather than the 2-norm
for Exercise 3).

# Exercise 1

Most people were able to implement the Exercise 1 code perfectly. One common
issue in the reports was that many people suggested that the numerical results
proved the Collatz conjecture (for example, because of the graph shape, or
because there were no sequences much above 200 in length). For theorems like
this, however, numerics can only find a counterexample. It's not possible to
*prove* a conjecture about all integers this way (at least, not unless your
experiment is part of a larger inductive argument).

Indeed, some conjectures in number theory have turned out to be false, but with
counterexamples that are much larger than any number representable on a
conventional computer. (For instance, see
[Skewes' number](https://en.wikipedia.org/wiki/Skewes'_number).)

# Exercise 2

Many people forgot to make their routines work in the case where the starting
value has a derivative of zero, but is not a minimum (for example, if the
starting point is a maximum).

One simple way of dealing with this is to use the function `f` that's passed in
to test values at a distance of `epsilon` on either side of the starting
point. If one of those values is smaller than the starting point, then the
iteration should "move" in that direction to find a true minimum. This is the
reason that `f` is an argument in the first place (as many of you noticed, we
don't use it for the gradient descent itself). This can also be done during the
iteration to avoid stopping at an inflection point (though we didn't test for
this).

Another common mistake was to move in the direction where `f` was smaller, but
then stop iteration. To actually find the minimum, it's not enough to avoid a
"bad" spot. You then have to continue to loop to find the minimum.

# Exercise 3

A common mistake was to simply forget to take the absolute value of the matrix
entries when checking whether a matrix is SDD.

A few people seem to have simply forgotten to use `gauss_seidel_step` in
`gauss_iteration` at all (due to copying-and-pasting code from `jacobi_step`;
some people had two calls to `jacobi_step`, but only changed one of them).

For the report, there was a bit of a hint that the behavior of the Jacobi and
Gauss-Seidel methods might change when `epsilon` was somewhere near 10^-16 (near
machine precision). If you look closely at what happens to the residual with
small values of `epsilon`, you might notice two things:

1. If `epsilon` is small enough, the number of operations that either method
   takes to converge stays constant. E.g., no matter how small `epsilon` is, the
   Jacobi method will stay well under 200 iterations for the problem given.

   This is because of limited precision of floating-point arithmetic. Eventually
   the difference between two iterations will be rounded to zero. So no matter
   what (positive) value of `epsilon` is used, the iteration will not continue
   past that point. (It is theoretically possible that the method could instead
   oscillate between two values, so that termination *never* happens if
   `epsilon` is too small. However, that does not happen in this case; if the
   method is stable, it is not common to enter an infinite loop with the kind of
   criterion we're using here.)

2. At the same time, as the number of iterations increases, the value of the
   residual eventually levels off, rather than decreasing to zero. This is
   because there's a small amount of truncation error that affects the
   calculated residual, and so the residual can't shrink below the order of
   magnitude of that error. This is easier to see if you plot the residual
   against the iteration count for both methods using a log scale (i.e. using
   the `semilogy` function from `matplotlib`).

   (This means that if our convergence criterion was based on the residual, we
   *would* enter an infinite loop if we tried to make the norm of the residual
   on the order of machine precision or smaller.)
