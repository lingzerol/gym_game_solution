import numpy as np
solution = np.array([0.5, 0.1, -0.3])
def f(w): return -np.sum((w - solution)**2)

npop = 50      # population size
sigma = 0.1    # noise standard deviation
alpha = 0.001  # learning rate
w = np.random.randn(3) # initial guess
for i in range(300):
  N = np.random.randn(npop, 3)
  R = np.zeros(npop)
  for j in range(npop):
    w_try = w + sigma*N[j]
    R[j] = f(w_try)
  A = (R - np.mean(R)) / np.std(R)
  w = w + alpha/(npop*sigma) * np.dot(N.T, A)


class evolution_strategy:
    """
    using evloution strategy to fit the model and predit action from different states

    parameters:
    npop - the population size
    sigma - noise standard deviation
    alpha - learning rate
    w - 2D-array, solution of different states, w[n_states][n_actions]
    """

    def __init__(self,env,npop=50,sigma=0.1,alpha=0.001):
        self.env=env
        self.npop=npop
        self.sigma=sigma
        self.alpha=alpha