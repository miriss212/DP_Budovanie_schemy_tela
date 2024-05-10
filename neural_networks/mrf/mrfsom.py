
try:
    from .util import *
except Exception: #ImportError
    from util import *


class MRFSOM:
    """
    MRF-SOM, Alg. for visualization was copied from:
    Neural Networks (2-AIN-132/15), FMFI UK BA
    (c) Tomas Kuzma, 2017-2018
    """

    def __init__(self, dim_in, n_rows, n_cols, mask=None, inputs=None):
        self.dim_in = dim_in
        self.n_rows = n_rows
        self.n_cols = n_cols
        if mask is None:
            self.mask = np.ones((n_rows, n_cols, dim_in))
        else:
            self.mask = mask

        self.weights = np.ones((n_rows, n_cols, dim_in))*0.5

        if inputs is not None:
            low  = np.min(inputs, axis=1)
            high = np.max(inputs, axis=1)
            self.weights = low + self.weights * 2 * (high - low)
        self.weights *= self.mask

        self.weights = self.mask * 0.5

    def distances(self, x):
        D = np.linalg.norm(x - self.weights, axis=2)
        return D.flatten().tolist()

    def winner(self, x):
        print(f"{self.weights.shape} x {x.shape}")
        D = self.weights @ x
        return np.unravel_index(np.argmax(D), D.shape)

    def activation(self, x):
        D = self.weights @ x
        return D.flatten().tolist()

    def winnerVector(self, x):
        #print(self.winner(x))
        r, c = self.winner(x)
        return self._toOneHot(r,c, self.n_rows, self.n_cols)

    def getAllNeighborsOfWinnerVector(self, x):
        r,c = self.winner(x)
        neighbor1 = min(0,r-1),c
        neighbor2 = max(r+1, self.n_rows),c
        neighbor3 = r, min(0,c-1)
        neighbor3 = r, max(c+1, self.n_cols)
        return [
            self._toOneHot(r, c, self.n_rows, self.n_cols),
            self._toOneHot(min(0,r-1),c, self.n_rows, self.n_cols),
            self._toOneHot(max(r+1, self.n_rows),c, self.n_rows, self.n_cols),
            self._toOneHot(r, min(0,c-1), self.n_rows, self.n_cols),
            self._toOneHot(r, max(c+1, self.n_cols), self.n_rows, self.n_cols),

            #cross-neighbors
            self._toOneHot(max(r+1, self.n_rows), max(c + 1, self.n_cols), self.n_rows, self.n_cols),
            self._toOneHot(min(0,r-1), max(c + 1, self.n_cols), self.n_rows, self.n_cols),
            self._toOneHot(max(r+1, self.n_rows), max(c - 1, self.n_cols), self.n_rows, self.n_cols),
            self._toOneHot(min(0,r-1), max(c - 1, self.n_cols), self.n_rows, self.n_cols),
        ]

    def _toOneHot(self, x, y, maxI, maxJ):
        res = []
        for i in range(maxI):
            for j in range(maxJ):
                if x == i and y == j:
                    res.append(1)
                else:
                    res.append(0)
        return res

    def train(self, inputs, metric, alpha_s=0.01, alpha_f=0.001, lambda_s=None, lambda_f=1, eps=10, trace=False, trace_interval=10):
        (_, count) = inputs.shape

        if trace:
            ion()
            plot_grid_3d(inputs, self.weights, block=False)
            redraw()

        for ep in range(int(eps)):

            alpha_t  = alpha_s  * (alpha_f/alpha_s)   ** ((ep)/(eps-1))
            lambda_t = lambda_s * (lambda_f/lambda_s) ** ((ep)/(eps-1))

            print()
            print('Ep {:3d}/{:3d}:'.format(ep+1,eps))
            print('  alpha_t = {:.3f}, lambda_t = {:.3f}'.format(alpha_t, lambda_t))

            for i in np.random.permutation(count):
                x = inputs[:,i]

                win_r, win_c = self.winner(x)

                C, R = np.meshgrid(range(self.n_cols), range(self.n_rows))
                D = metric(np.stack((R, C)), np.reshape((win_r, win_c), (2,1,1)))
                Q = np.exp(-(D/lambda_t)**2)

                tmp = self.weights + alpha_t * np.atleast_3d(Q) * x
                tmp = self.mask * tmp
                tmpNorm = np.reshape(np.linalg.norm(tmp, axis=2), (self.n_rows, self.n_cols, 1))+0.00001
                self.weights = tmp / tmpNorm

            if trace and ((ep+1) % trace_interval == 0):
                plot_grid_3d(inputs, self.weights, block=False)
                redraw()

        if trace:
            ioff()
