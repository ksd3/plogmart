import numpy as np

def chi_squared(y, A, x, sigma):
    """
    Calculate the chi-squared value between the observed data and model prediction.

    Parameters:
    - y: NumPy array (Nx1), the observed data vector.
    - A: NumPy array (NxM), the system matrix.
    - x: NumPy array (Mx1), the model prediction vector.
    - sigma: float, standard deviation of the observed data.

    Returns:
    - float, the chi-squared value.
    """
    return np.sqrt(np.sum(((A @ x - y) / sigma) ** 2))

def logmart(y, A, relax=20, x0=None, sigma=1, max_iter=20):
    """
    Solve y=Ax using parallel log-entropy MART (Modified ART) algorithm with chi-squared stopping criterion.

    Parameters:
    - y (NumPy array, Nx1): The observed data vector. Must be nonnegative. Small values are thresholded to avoid division by zero.
    - A (NumPy array, NxM): The system matrix. Must be nonnegative.
    - relax (float, optional): Relaxation constant to control the update magnitude. Defaults to 20.
    - x0 (NumPy array, optional): Initial guess for the solution (Mx1 vector). If None, an initial guess is computed. Defaults to None.
    - sigma (float, optional): Standard deviation of the observed data, used in chi-squared calculation. Defaults to 1.
    - max_iter (int, optional): Maximum number of iterations to perform. Defaults to 20.

    Returns:
    - x (NumPy array, Mx1): The solution vector.
    - y_est (NumPy array, Nx1): The estimated data vector from the final solution.
    - chi2 (float): The final chi-squared value.
    - iter_count (int): The number of iterations performed.
    """
    # Ensure y has no zeros to avoid division by zero in calculations.
    y = np.maximum(y, 1e-8)

    # Initialize the solution vector either with a provided initial guess or a computed initial guess.
    if x0 is None:
        x = (A.T @ y) / A.sum()
        x *= np.max(y) / np.max(A @ x)
    else:
        x = x0

    # Weight vector for the MART algorithm, initialized to uniform weights normalized by the number of observations.
    W = np.ones(A.shape[0]) / A.shape[0]

    # Initial chi-squared value for the starting solution.
    chi2 = chi_squared(y, A, x, sigma)

    # Backup solution vector in case the update increases the chi-squared value.
    xold = x.copy()

    # Perform iterative solution update until the maximum iteration count or chi-squared stopping criterion is met.
    for i in range(max_iter):
        # Compute the MART update factor.
        t = np.min(1 / (A @ x))
        C = relax * t * (1 - ((A @ x) / y))

        # Update the solution vector.
        x /= (1 - x * (A.T @ (W * C)))

        # Check for chi-squared stopping criterion.
        chiold = chi2
        chi2 = chi_squared(y, A, x, sigma)

        if chi2 > chiold and i > 2:
            x = xold  # Revert to previous solution if chi-squared value increases.
            break
        else:
            xold = x.copy()  # Update the backup solution.

    # Compute the estimated data vector from the final solution.
    y_est = A @ x

    # Return the final solution and diagnostics.
    iter_count = i + 1
    return x, y_est, chi2, iter_count
