import numpy as np
from itertools import chain, combinations
def powersetiter(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def tor(M):
    """ Compute the torontonian of matrix.
    This matrix has certain constraints, as it is related via similarity to a Covariance matrix
    See https://arxiv.org/abs/1807.01639 for details
    """
    m = len(M)
    assert(M.shape == (m,m))
    assert(m%2 == 0)
    n = m//2
    ssum = 0.0
    for i in powersetiter(range(n)):
        ia = np.array(i)
        ii = list(np.concatenate((ia,ia+n),axis=0))
        Ms = np.delete(M,ii,axis=0)
        Ms = np.delete(Ms,ii,axis=1)
        ll = len(Ms)
        if ll != 0: #Check it is not the "empty matrix"
            dd = np.linalg.det(np.identity(ll)-Ms).real
        else:
            dd = 1
        ssum += (-1)**(len(i))/np.sqrt(dd)
    return ssum
