def collatz_step(n):
    """Returns the result of the Collatz function.

    The Collatz function C : N -> N is used in `collatz` to
    generate collatz sequences.

    Parameters
    ----------
    n : int

    Returns
    -------
    int
    """
    if (n < 1):
        raise ValueError('n must be >= 1')

    if (n == 1):
        return 1

    if (n % 2 == 0):
        return n/2
    elif (n % 2 == 1):
        return 3*n + 1

def collatz(n):
    """Returns the Collatz sequence beginning with `n`.

    It is conjectured that Collatz sequences all end with `1`.

    Parameters
    ----------
    n : int

    Returns
    -------
    sequence : list
        A Collatz sequence.
    """
    sequence = [n]
    while (n > 1):
        n = collatz_step(n)
        sequence.append(n)
    return sequence
