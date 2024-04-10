import numpy as np
import pickle
import os

try:
    from .util import *
except Exception: #ImportError
    from util import *

class SOMLoader:

    def __init__(self):
        pass

    def load_som(self, dim, rows, cols, folderName):
        model = SOM(dim, rows, cols)
        for i in range(rows):
            model.weights[i] = np.loadtxt(folderName + "/" + str(i) + ".txt")
        return model

class SOMSaver:

    def __init__(self):
        pass

    def save_som(self, som, name):
        # try:
        #     os.mkdir("./data/"+name+"/")
        # except OSError:
        #     pass

        for i in range(som.weights.shape[0]): 
            file_name = "data/trained/som/" + name + "/" + str(i) + ".txt"
            f = open(file_name, "w")
            np.savetxt(file_name, som.weights[i])
            f.close()


class SOM:
    """
        MRF-SOM, Alg. for visualization was copied from:
        Neural Networks (2-AIN-132/15), FMFI UK BA
        (c) Tomas Kuzma, 2017-2018
    """
    def __init__(self, dim_in, n_rows, n_cols, inputs=None):
        self.dim_in = dim_in
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.quant_error_array = []
        self.weights = np.ones((n_rows, n_cols, dim_in))*0.5
        self.winning_counts = {}

        if inputs is not None:
            self.data = inputs
            low  = np.min(inputs, axis=1)
            high = np.max(inputs, axis=1)
            self.weights = low + self.weights * 2 * (high - low)

    def distances(self, x):
        D = np.linalg.norm(x - self.weights, axis=2)
        return D.flatten().tolist()

    def winner(self, x):
        D = np.linalg.norm(x - self.weights, axis=2)
        # print(D)
        # print(D[np.unravel_index(np.argmin(D), D.shape)])
        return np.unravel_index(np.argmin(D), D.shape)

    def winnerVector(self, x):
        # print("original")
        r, c = self.winner(x)
        # print(r," ",c)
        return self._toOneHot(r,c, self.n_rows, self.n_cols)

    def _toOneHot(self, x, y, maxI, maxJ):
        res = []
        for i in range(maxI):
            for j in range(maxJ):
                if x == i and y == j:
                    res.append(1)
                else:
                    res.append(0)
        return res

    def fromOneHot(self, one_hot_vector):

        current_row = 0
        for i in range(len(one_hot_vector)):
            if i > 0 and i % self.n_rows == 0:
                current_row += 1

            current_col = i % self.n_cols

            if one_hot_vector[i] == 1:
                return current_row, current_col

        return 0,0


        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if one_hot_vector[row+col] == 1:
                    return row, col
        raise Exception("ERROR: OneHot not read correctly!")

    def smallest_indices(self, ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, n)[:n]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    def k_winner_vector(self, x, k=4):
        D = np.linalg.norm(x - self.weights, axis=2)
        # print(D)
        win_coords = self.smallest_indices(D, k)
        result = np.zeros((self.n_rows, self.n_cols))
        result[win_coords] = 1.0
        coords_x, coords_y = win_coords
        r = coords_x[0]
        c = coords_y[0]
        # print("kwta")
        # print(r, " ", c)
        # print(D[r][c])
        return result.flatten().tolist()

    def k_winners_conti(self, x, k=12):
        D = np.linalg.norm(x - self.weights, axis=2)
        win_coords = self.smallest_indices(D, k)
        result = np.zeros((self.n_rows, self.n_cols))
        #print(D[win_coords])
        coords_x, coords_y = win_coords
        winner = (coords_x[0], coords_y[0])
        rest = (coords_x[1:], coords_y[1:])
        result[winner] = 1.0
        result[rest] = 0.5
        #print(result)
        return result.flatten().tolist()

    #def train(self, inputs, metric=lambda u,v:0, alpha_s=0.01, alpha_f=0.001, lambda_s=None, lambda_f=1, eps=100, trace=False, trace_interval=10):
    def train(self, inputs, metric=lambda u,v:0, alpha_s=0.01, alpha_f=0.01, lambda_s=None, lambda_f=1, eps=10, trace=True, trace_interval=10):
        (_, count) = inputs.shape

        
        """if trace:
            ion()
            plot_grid_3d(inputs, self.weights, block=False)
            redraw()"""
        """timestamp = int(time.time())
        directory = f'data/{timestamp}/'
        if not os.path.exists(directory):
            os.makedirs(directory)"""
        for i in range(self.n_rows):
                for j in range(self.n_cols):
                    self.winning_counts[(i, j)] = 0 


        for ep in range(eps):
            #tempo ucenia
            alpha_t  = alpha_s  * (alpha_f/alpha_s)   ** ((ep)/(eps-1)) #Kontroluje, do akej miery sa váhy neurónov aktualizujú počas trénovania. Čím väčšie je alpha, 
            #tým väčší je vplyv každého vstupu na aktualizáciu váh neurónov.
            #polomer susedstva
            lambda_t = lambda_s * (lambda_f/lambda_s) ** ((ep)/(eps-1)) #Čím väčšie je lambda, tým väčší je polomer susedstva
            #a teda aj väčšia oblasť ovplyvnenia susedných neurónov.

            print()
            print('Ep {:3d}/{:3d}:'.format(ep+1,eps))
            print('  alpha_t = {:.3f}, lambda_t = {:.3f}'.format(alpha_t, lambda_t))

            for i in range(count):
                x = inputs[:, i]
                win_r, win_c = self.winner(x)

                # Update winning count for the winning neuron
                self.winning_counts[(win_r, win_c)] += 1

            for i in np.random.permutation(count):
                x = inputs[:,i]
                win_r, win_c = self.winner(x)

                C, R = np.meshgrid(range(self.n_cols), range(self.n_rows))
                D = metric(np.stack((R, C)), np.reshape((win_r, win_c), (2,1,1)))
                Q = np.exp(-(D/lambda_t)**2)
                self.weights += alpha_t * np.atleast_3d(Q) * (x - self.weights)
            
            quant_err = self.quant_err(inputs)  # Vypocet quantization erroru
            self.quant_error_array.append(quant_err)  # Ulozit quantization error do pola

            if trace and ((ep+1) % trace_interval == 0):
                plot_grid_2d(inputs, self.weights, block=False)
                redraw()

        # Print winning counts at the end of training
        print("Winning counts for each neuron:")
        for neuron, count in self.winning_counts.items():
            print(f"Neuron {neuron}: {count} times")
            


        if trace:
            ioff()


    def plot_map(self):
        plot_grid_2d(self.data, self.weights, block=False)
        plt.colorbar()
        plt.show()

    def quant_err(self, data=None):
        """
            Computes the quantization error of the SOM.
            It uses the data fed at last training (optionally, the supplied data if not None).
        """
        if data is not None and data.any():
            data_to_check = data
        else :
            data_to_check = self.data
        dists = []
        (_, count) = data_to_check.shape
        for i in np.random.permutation(count):
            input_vector = data_to_check[:, i]
            #print("input vector: ", input_vector)
            # winner = self.winnerVector(input_vector)
            # dists.append(np.linalg.norm(input_vector - winner))
            win_r, win_c = self.winner(input_vector)
            winner = self.weights[win_r][win_c]
            #print("Winner is ")
            #print(winner)
            dists.append(np.linalg.norm(input_vector - winner))

        return np.array(dists).mean()
    
    def plot_winner_histogram(som, title="Winner Histogram"):
        # Extract the winning counts
        neurons = [f'Neuron {i}' for i in range(len(som.winning_counts))]
        wins = list(som.winning_counts.values())

        # Plot the histogram
        plt.bar(neurons, wins, align='center', edgecolor='black')
        plt.xlabel('Neurons')
        plt.ylabel('Number of Wins')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.show()

    def winner_diff(self, data=None):
       
        if data is not None and data.any():
            data_to_check = data
        else:
            data_to_check = self.data

        winner_sets = [] 

        (_, count) = data_to_check.shape

        for i in np.random.permutation(count): #nepotrebujem random permutation, poradie v ktorom to tam strkam by nemalo ovplyvnit vysledok
            input_vector = data_to_check[:, i]
            win_r, win_c = self.winner(input_vector)
            winner_sets.append((win_r, win_c))

        all_winners_set = set(winner_sets)
        num_all_winners = len(all_winners_set)

        winner_diff = num_all_winners / (self.n_rows * self.n_cols)

        return winner_diff
    
    def compute_entropy(self, data=None):
        if data is not None and data.any():
            data_to_check = data
        else:
            data_to_check = self.data

        (_, count) = data_to_check.shape
        entropy = 0.0

        for i in np.random.permutation(count):
            input_vector = data_to_check[:, i]
            win_r, win_c = self.winner(input_vector)
            wins_for_neuron = 1 if self.fromOneHot(self.winnerVector(input_vector)) == (win_r, win_c) else 0
            entropy_term = wins_for_neuron / count * np.log(wins_for_neuron / count + 1e-10)
            entropy += entropy_term

        # vynasobime -1, taka je konvencia 
        return -entropy


    








    """
        Computes the winner differentiation of the SOM.
        It uses the data fed at last training (optionally, the supplied data if not None).

        Returns:
        winner_diff (float): Average winner differentiation.
        """

    """
    def winner_diff(self, data=None):
        
        if data is not None and data.any():
            data_to_check = data
        else:
            data_to_check = self.data

        winner_diffs = []
        (_, count) = data_to_check.shape

        # Save the original weights
        original_weights = np.copy(self.weights)

        for i in np.random.permutation(count):
            input_vector = data_to_check[:, i]
            win_r, win_c = self.winner(input_vector)
            winner_vector = self._toOneHot(win_r, win_c, self.n_rows, self.n_cols)

            # Save the original winner coordinates
            original_win_r, original_win_c = win_r, win_c

            # Determine the new winner without updating the weights
            new_win_r, new_win_c = self.winner(input_vector)
            new_winner_vector = self._toOneHot(new_win_r, new_win_c, self.n_rows, self.n_cols)

            differentiation = np.sum(np.abs(np.subtract(new_winner_vector, winner_vector)))
            winner_diffs.append(differentiation)

            # Restore the original winner coordinates for the next iteration
            win_r, win_c = original_win_r, original_win_c

        # Restore the original weights
        self.weights = original_weights

        return np.mean(winner_diffs)
        """