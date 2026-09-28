import math


def my_pi(target_error):
    """
    Implementation of Gauss–Legendre algorithm to approximate PI from https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_algorithm

    :param target_error: Desired error for PI estimation
    :return: Approximation of PI to specified error bound
    """

    ### YOUR CODE HERE ###
  # Initial values
    a = 1
    b = 1 / math.sqrt(2)
    p = 1
    t = 1 / 4

    while True:

        # Save the old values
        old_a = a
        old_b = b

        # Gauss-Legendre updates
        a = (old_a + old_b) / 2
        b = math.sqrt(old_a * old_b)
        t = t - p * (old_a - old_b) ** 2 / 4
        p = 2 * p

        # Calculate approximation of pi
        approximation = (a + b) ** 2 / (4 * t)

        # Check whether desired error has been reached
        if abs(math.pi - approximation) < target_error:
            return approximation






desired_error = 1E-10

approximation = my_pi(desired_error)

print("Solution returned PI=", approximation)

error = abs(math.pi - approximation)

if error < abs(desired_error):
    print("Solution is acceptable")
else:
    print("Solution is not acceptable")
