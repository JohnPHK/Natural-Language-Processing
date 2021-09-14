from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

dataDir = "/u/cs401/A3/data/"


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))
        self.precomp_term = np.zeros(M) 
        self.precomp_ind = 0 # 0 meaning precomputation not performed for the new parameters. 1 meaning precomutation performed for the new parameters.

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        # print(self.precomp_term.shape)
        return self.precomp_term[m]

    def reset_precomp_term(self):
        term_1 = (0.5 * ((self.mu ** 2) / self.Sigma)).sum(axis=1)
        term_2 = (0.5 * self._d) * np.log(2 * np.pi) 
        term_3 = 0.5 * np.log(self.Sigma).sum(axis=1) 
        precomp_term = term_1 + term_2 + term_3
        assert precomp_term.size == self._M
        self.precomp_term = precomp_term 
        self.precomp_ind =1

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    if myTheta.precomp_ind == 0:
        myTheta.reset_precomp_term()
    s = myTheta.Sigma[m][np.newaxis, :]
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    first_term = - ((0.5 * (np.einsum('ij,ij->i', x/s, x))) - 
                    (np.einsum('ij,ij->i', myTheta.mu[m]/s, x)))
    to_be_returned = first_term - myTheta.precomputedForM(m)
    if len(to_be_returned.shape) == 1:
        if to_be_returned.shape[0] == 1:
            return to_be_returned[0] 
    return to_be_returned


def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    sum_ob = log_Bs + np.log(myTheta.omega) #Also the numerator
    # a = sum_ob.max(axis=0, keepdims=True)
    a = np.max(sum_ob, axis=0, keepdims=True)
    denom = a + logsumexp(sum_ob - a, axis=0, keepdims=True) 
    return sum_ob - denom


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    sum_ob = log_Bs + np.log(myTheta.omega)
    a = np.max(sum_ob, axis=0, keepdims=True)
    return np.sum(a + logsumexp(sum_ob-a, axis=0, keepdims=True)) 


def compute_log_llh(X, M, Theta): 
    log_bmxs = np.array([log_b_m_x(m, X, Theta) for m in range(M)])
    log_pxs = log_p_m_x(log_bmxs, Theta) 
    return logLik(log_bmxs, Theta), log_pxs 


def update_parameters(Theta, X, log_pxs):
    num_pms = np.exp(log_pxs)
    a = log_pxs.max(1, keepdims=True)
    denom_pms = np.exp(a + logsumexp(log_pxs - a, axis=1, keepdims=True))
    
    # reset omega
    Theta.reset_omega(np.mean(num_pms, 1))

    # reset mu
    Theta.reset_mu((num_pms @ X) / denom_pms)
    
    # reset sigma
    first_term = (num_pms @ np.power(X, 2)) / denom_pms
    second_term = 1e-9 - np.power(Theta.mu, 2)
    Theta.reset_Sigma(first_term + second_term) 
    Theta.precomp_ind = 0


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    # perform initialization (Slide 32)
    # print("TODO : Initialization")
    # for ex.,
    # myTheta.reset_omega(omegas_with_constraints)
    # myTheta.reset_mu(mu_computed_using_data)
    # myTheta.reset_Sigma(some_appropriate_sigma)

    # Mu initialization
    random_data = X[np.random.randint(0, X.shape[0], M)]
    myTheta.reset_mu(random_data)

    # Sigma initialization
    sigma = np.ones((M, X.shape[1]))
    myTheta.reset_Sigma(sigma)

    # Omega initialization
    omega = np.ones((M, 1))/M
    myTheta.reset_omega(omega) 

    # Indicate we need to precompute with initialized parameters.
    myTheta.precomp_ind = 0

    # print("TODO: Rest of training")
    i = 0
    prev_l = -np.inf
    improvement = np.inf
    while i < maxIter and improvement >= epsilon:
        myTheta.reset_precomp_term()
        
        # Compute loglikelihood.
        l, log_pxs = compute_log_llh(X, M, myTheta)

        # Update parameters.
        update_parameters(myTheta, X, log_pxs)

        # Update variables
        improvement = abs(l - prev_l)
        prev_l = l
        i += 1
    myTheta.reset_precomp_term()
    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    bestModel = -1
    llhs = []
    i = 0
    for model in models:
        log_bmxs = np.array([log_b_m_x(m, mfcc, model) for m in range(model._M)]) 
        llhs.append((i, logLik(log_bmxs, model), model))
        i += 1

    llhs = sorted(llhs, key=lambda tup: tup[1], reverse=True)
    bestModel = llhs[0][0]

    print(models[correctID].name)
    if k > 0:
        for i in range(k):
            print(f"{llhs[i][2].name} {llhs[i][1]}")
        print("")
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    # print("TODO: you will need to modify this main block for Sec 2.3")
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8 
    epsilon = 0
    maxIter = 20 
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    #### Below code used for generating only subset of data ####
    # subset_speakers = ['S-10B', 'S-21A', 'S-7C', 'S-19C', 'S-27C', 'S-23C', 'S-6B', 'S-5A', 'S-11C', 'S-16D', 'S-20D', 'S-29A', 'S-14B', 'S-32D', 'S-22B', 'S-28D', 'S-4D', 'S-8D', 'S-13A', 'S-3C', 'S-12D', 'S-18B', 'S-26B', 'S-30B', 'S-9A', 'S-2B', 'S-17A', 'S-25A', 'S-15C', 'S-1A', 'S-24D', 'S-31C']
    # random.shuffle(subset_speakers)
    # for _ in range(4):
        # subset_speakers.pop()

    # print(subset_speakers)
    # print(len(subset_speakers))
 

    # for speaker in subset_speakers:

        # files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
        # random.shuffle(files)

        # testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
        # testMFCCs.append(testMFCC)

        # X = np.empty((0, d))

        # for file in files:
            # myMFCC = np.load(os.path.join(dataDir, speaker, file))
            # X = np.append(X, myMFCC, axis=0)

        # trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    #### Above code used for generating only subset of data ####

    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print(f"accuracy: {accuracy}")
