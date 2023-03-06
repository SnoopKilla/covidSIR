from src.utility_functions import log_pi_lambda, log_pi_t
import numpy as np


def update_lambda(lam, t, s, i, P, sigma, alpha, beta, phi):
    # This function updates the parameter vector lambda.
    # INPUT:
    # - lam: array of the values of lambda;
    # - t: array of the breakpoints;
    # - s: array of susceptible individuals during time;
    # - i: array of infected individuals during time;
    # - P: total number of individuals;
    # - sigma: algorithm parameter for the proposal of a new candidate lambda;
    # - alpha, beta: hyperparameters of the prior of lambda;
    # - phi parameter phi of the model.
    # OUTPUT:
    # - lam: update array;
    # - accept: number of acceted candidates.
    # NOTES: The update is done component-wise in a sequential manner.

    current = np.copy(lam)  # Get the current state of the chain.
    candidate = np.copy(current)  # Initialize the new candidate.

    # For every component of the parameter vector, we tweak such component
    # according to the chosen proposal and then update the chain according
    # to the computed acceptance rate.
    accepted = 0  # Initialize the count of accepted candidates.
    for j in range(current.shape[0]):
        # Tweak the j-th component.
        candidate[j] = candidate[j] + sigma * np.random.normal()
        # Compute the acceptance rate.
        log_alpha = (log_pi_lambda(candidate, t, s, i, P, alpha, beta, phi)
                     - log_pi_lambda(current, t, s, i, P, alpha, beta, phi))

        # If the candidate is accepted, we move the chain (current = candidate)
        # and increase the count of accepted candidates. Otherwise, we reject
        # the candidate and the chain does not move from the current state
        # (candidate = current).
        if log_alpha > np.log(np.random.uniform()):
            current = np.copy(candidate)
            accepted = accepted + 1
        else:
            candidate = np.copy(current)

    return current.reshape(-1, 1), accepted


def update_t(lam, t, s, i, P, M, phi):
    # This function updates the parameter vector lambda.
    # INPUT:
    # - lam: array of the values of lambda;
    # - t: array of the breakpoints;
    # - s: array of susceptible individuals during time;
    # - i: array of infected individuals during time;
    # - P: total number of individuals;
    # - M: algorithm parameter for the proposal of a new candidate t;
    # - phi parameter phi of the model.
    # OUTPUT:
    # - lam: update array;
    # - accept: number of acceted candidates.
    # NOTES: The update is done component-wise in a sequential manner.

    current = np.copy(t)  # Get the current state of the chain.
    candidate = np.copy(current)  # Initialize the new candidate.

    # For every component of the parameter vector, we tweak such component
    # according to the chosen proposal and then update the chain according
    # to the computed acceptance rate.
    accepted = 0  # Initialize the count of accepted candidates.
    for j in range(current.shape[0]):
        # Tweak the j-th component.
        candidate[j] = candidate[j] + np.random.choice(np.arange(-M, M + 1))
        # Compute the acceptance rate.
        log_alpha = (log_pi_t(lam, candidate, s, i, P, phi)
                     - log_pi_t(lam, current, s, i, P, phi))

        # If the candidate is accepted, we move the chain (current = candidate)
        # and increase the count of accepted candidates. Otherwise, we reject
        # the candidate and the chain does not move from the current state
        # (candidate = current).
        if log_alpha > np.log(np.random.uniform()):
            current = np.copy(candidate)
            accepted = accepted + 1
        else:
            candidate = np.copy(current)

    return current.reshape(-1, 1), accepted


def mcmc_sampler(s, i, d, P, n_iterations, burnin, M, sigma,
                 alpha, beta, a, b, phi):
    # This function implement the hybrid MCMC sampler.
    # INPUT:
    # - s: array of susceptible individuals during time;
    # - i: array of infected individuals during time;
    # - d: number of breakpoints;
    # - P: total number of individuals;
    # - n_iterations: number of iterations for the algorithm;
    # - burnin: number of burnin iterations to discard;
    # - M: algorithm parameter for the proposal of a new candidate t;
    # - sigma: algorithm parameter for the proposal of a new candidate lambda;
    # - alpha, beta: hyperparameters of the prior of lambda;
    # - a, b: hyperparameters of the prior of p;
    # - phi: parameter phi of the model.
    # OUTPUT:
    # - p: simulated chain for the probability of removal from
    # infected population;
    # - lam: simulated chain for lambda;
    # - t: simulated chain for the breakpoints.

    T = s.shape[0] - 1  # Index of the final time instant.

    # Initialize the parameters.

    # The initial value of p is drawn from the prior distribution.
    p = np.random.beta(a, b, size=(1, 1))
    # Each of the d breakpoints (t_i) is drawn randomly (without replacement)
    # between 1 and T-1. The obtained vector is then sorted to make sure
    # that t_1 < t_2 < ... < t_d.
    t = np.sort(np.random.choice(np.arange(1, T), size=d-1, replace=False))
    t = t.reshape(-1, 1)
    # Each of the lambda_i's is drawn independently from
    # its prior distribution.
    lam = np.random.gamma(alpha, beta)
    lam = lam.reshape(-1, 1)

    # Compute the hyperparameters of the posterior of p.
    a_new = a + i[0] - i[-1] + s[0] - s[-1]
    b_new = b + np.sum(i[1:]) + s[-1] - s[0]

    # Initialize the count of accepted candidates for lambda and t.
    a_lam = 0
    a_t = 0

    # Run the chain
    for _ in range(n_iterations):
        # Update p by sampling from its posterior.
        p = np.hstack((p, np.random.beta(a_new, b_new, size=(1, 1))))
        # Update lam via Metropolis-Hastings step.
        new_lam, accepted_lam = update_lambda(lam[:, -1], t[:, -1], s, i, P,
                                              sigma, alpha, beta, phi)
        # Update t via Metropolis-Hastings step.
        new_t, accepted_t = update_t(lam[:, -1], t[:, -1], s, i, P, M, phi)
        lam = np.hstack((lam, new_lam))
        t = np.hstack((t, new_t))

        # Update the counts of accepted candidates for lambda and t.
        a_lam = a_lam + accepted_lam
        a_t = a_t + accepted_t

    # Compute the acceptance rates for lambda and t.
    lam_ar = a_lam / n_iterations / d
    t_ar = a_t / n_iterations / (d-1)

    # Discard burn-in iterations.
    p = p[:, burnin:]
    lam = lam[:, burnin:]
    t = t[:, burnin:]

    return p, lam, t, lam_ar, t_ar
