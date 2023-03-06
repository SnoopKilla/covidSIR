import numpy as np
from scipy.special import gammaln


def lambda_time(lam, t, time):
    # This function computes the value of lambda.
    # INPUT:
    # - lam, t: arrays defining the (piecewise constant) function lambda(t);
    # - time: time instants at which we want to evaluate lambda.
    # OUTPUT:
    # - lambda_time: value of lambda.
    # NOTES: The function is vectorized in the array time. Indeed,
    # it allows to compute kappa for all the time instants in the array time.

    lambda_time = lam[np.searchsorted(t, time, side="right")]
    return lambda_time


def compute_kappa_time(s_t, i_t, lam, t, time, phi, P):
    # This function computes the value of kappa.
    # INPUT:
    # - s_t: number of susceptible individuals;
    # - i_t: number of infected individuals;
    # - lam, t: arrays defining the (piecewise constant) function lambda(t);
    # - time: time instants at which we want to evaluate kappa;
    # - phi: parameter phi of the model;
    # - P: total number of individuals.
    # OUTPUT:
    # - kappa_time: value of kappa.
    # NOTES: The function is vectorized in the arrays time, s_t and i_t.
    # Indeed, it allows to compute kappa for all the time instants in
    # the array time. It is required that time, s_t and i_t
    # have the same dimension.

    p_si_t = 1 - np.exp(-np.multiply(lambda_time(lam, t, time), i_t) / P)
    kappa_time = (1/phi - 1) * np.multiply(s_t, p_si_t)
    return kappa_time


def log_pi_lambda(lam, t, s, i, P, alpha, beta, phi):
    # This function computes the (log-) full-conditional
    # of the parameter vector lambda.
    # INPUT:
    # - lam: array of the values of lambda;
    # - t: array of the breakpoints;
    # - s: array of susceptible individuals during time;
    # - i: array of infected individuals during time;
    # - P: total number of individuals;
    # - alpha, beta: hyperparameters of the prior of lambda;
    # - phi: parameter phi of the model.
    # OUTPUT:
    # - result: (log-) full-conditional of lambda evaluated at lam.

    T = s.shape[0] - 1  # Index of the final time instant.
    time = np.arange(T + 1)  # Array of all time instants.

    # First, we initialize the result to -inf. If all the components of the
    # vector lam are positive (i.e., the vector is admissible) we compute the
    # (log-) full-conditional of lambda evaluated at lam and return the result.
    result = np.NINF
    if all(lam > 0):
        kappa_vec = compute_kappa_time(s[:-1], i[:-1], lam,
                                       t, time[:-1], phi, P)
        result = (np.sum(gammaln(-np.diff(s) + kappa_vec)
                         + kappa_vec * np.log(1 - phi)
                         - gammaln(kappa_vec))
                  + np.sum(np.log(np.power(lam, alpha-1))
                           - np.multiply(beta, lam)))
    return result


def log_pi_t(lam, t, s, i, P, phi):
    # This function computes the (log-) full-conditional
    # of the parameter vector t.
    # INPUT:
    # - lam: array of the values of lambda;
    # - t: array of the breakpoints;
    # - s: array of susceptible individuals during time;
    # - i: array of infected individuals during time;
    # - P: total number of individuals;
    # - phi parameter phi of the model.
    # OUTPUT:
    # - result: (log-) full-conditional of t evaluated at t.

    T = s.shape[0] - 1  # Index of the final time instant.
    time = np.arange(T + 1)  # Array of all time instants.

    # First, we initialize the result to -inf. If we have that
    # 0 < t_1 < t_2 < ... < t_(d-1) < T (i.e., the vector is admissible)
    # we compute the (log-) full-conditional of t evaluated at t
    # and return the result.
    result = np.NINF
    if np.all(np.diff(t) > 0) and t[0] > 0 and t[-1] < T:
        kappa_vec = compute_kappa_time(s[:-1], i[:-1], lam,
                                       t, time[:-1], phi, P)
        result = np.sum(gammaln(-np.diff(s) + kappa_vec)
                        + kappa_vec * np.log(1 - phi)
                        - gammaln(kappa_vec))
    return result


def simulate_data(T, lam, t, s_0, i_0, p_r, phi):
    # This function simulates data according to the process described above.
    # INPUT:
    # - T: index of the final time instant;
    # - lam, t: (true) arrays defining the function lambda(t);
    # - s_0: initial number of susceptible individuals;
    # - i_0: initial number of infected individuals;
    # - p_r: (true) probability of removal from infected population;
    # - phi: parameter phi of the model.
    # OUTPUT:
    # - s: array of susceptible individuals during time;
    # - i: array of infected individuals during time.

    # Compute the total number of individuals.
    P = s_0 + i_0

    # Initialize the arrays s and t.
    s = np.array([s_0])
    i = np.array([i_0])

    time = np.arange(T + 1)
    for t_i in time:
        # Draw a realization of delta_r.
        delta_r = np.random.binomial(i[-1], p_r)
        # Compute the kappa parameter at time t_i.
        kappa = compute_kappa_time(s[-1], i[-1], lam, t, t_i, phi, P)
        # Draw a realization of delta_i.
        delta_i = np.random.negative_binomial(kappa, 1 - phi)

        # Update s and i according to the model.
        s = np.append(s, s[-1] - delta_i)
        i = np.append(i, i[-1] + delta_i - delta_r)
    return s, i
