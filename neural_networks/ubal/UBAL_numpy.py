import numpy as np
from scipy.special import expit


class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, net):
        # return 1.0 / (1.0 + np.exp(-net))
        return expit(net)


class SoftMax:
    def __init__(self):
        pass

    def __call__(self, net):
        e_net = np.exp(net - np.max(net))
        e_denom0 = e_net.sum(axis=0, keepdims=True)
        result = e_net / e_denom0
        return result


class UBAL:
    def __init__(self, layers, act_fun_f, act_fun_b, learning_rate, init_w_mean, init_w_variance,
                 betas, gammas_f, gammas_b):
        super(UBAL, self).__init__()
        self.arch = layers
        self.d = len(self.arch)
        self.act_fun_F = act_fun_f
        self.act_fun_B = act_fun_b
        self.learning_rate = learning_rate
        self.init_weight_mean = init_w_mean
        self.init_weight_variance = init_w_variance
        self.betas = betas
        self.gammasF = gammas_f
        self.gammasB = gammas_b
        self.weightsF = []
        for i in range(self.d - 1):
            self.weightsF.append(np.random.normal(self.init_weight_mean, self.init_weight_variance,(self.arch[i+1], self.arch[i]+1)))
        self.weightsB = []
        for i in range(self.d - 1):
            self.weightsB.append(np.random.normal(self.init_weight_mean, self.init_weight_variance,(self.arch[i],self.arch[i+1]+1)))

    def add_bias(self, input_array):
        return np.vstack([input_array, np.ones(len(input_array[0]))])

    def activation(self, input_x, input_y):
        act_fp = [None] * self.d
        act_bp = [None] * self.d
        act_fe = [None] * self.d
        act_be = [None] * self.d

        act_fp[0] = input_x
        for i in range(1, self.d):
            act_fp[i] = self.act_fun_F[i](np.dot(self.weightsF[i-1], self.add_bias(act_fp[i-1])))
            act_fe[i-1] = self.act_fun_B[i](np.dot(self.weightsB[i-1], self.add_bias(act_fp[i])))

        act_bp[self.d-1] = input_y
        for i in range(self.d-1, 0, -1):
            act_bp[i-1] = self.act_fun_F[i](np.dot(self.weightsB[i-1], self.add_bias(act_bp[i])))
            act_be[i] = self.act_fun_B[i](np.dot(self.weightsF[i-1],self.add_bias(act_bp[i-1])))

        return act_fp, act_fe, act_bp, act_be

    def learning(self, act_fp, act_fe, act_bp, act_be):
        target = [None] * self.d
        target_with_bias = [None] * self.d
        estimateF = [None] * self.d
        estimateB = [None] * self.d

        for l in range(len(self.arch)):
            #print(f"beta : {self.betas[l]}")
            # t_q = beta^F_q q^FP + (1 - beta^F_q) q^BP
            target[l] = self.betas[l] * act_fp[l]  + (1.0 - self.betas[l]) * act_bp[l]
            target_with_bias[l] = self.add_bias(target[l])
            if l > 0:
                # e^F_q = gamma^F_q q^FP + (1 − gamma^F_q) q^BE
                estimateF[l] = self.gammasF[l] * act_fp[l] + (1.0 - self.gammasF[l]) * act_be[l]
            if l < (self.d - 1):
                # e^B_p = gamma^B p^BP + (1 - gamma^B_p) p^FE
                estimateB[l] = self.gammasB[l] * act_bp[l] + (1.0 - self.gammasB[l]) * act_fe[l]

        for l in range(self.d - 1):
            k = l + 1
            weight_update_F = self.learning_rate * np.dot(target_with_bias[l], (target[k] - estimateF[k]).transpose())
            weight_update_B = self.learning_rate * np.dot(target_with_bias[k], (target[l] - estimateB[l]).transpose())
            self.weightsF[l] += weight_update_F.transpose()
            self.weightsB[l] += weight_update_B.transpose()

    def activation_fp_last(self, input_x):
        act_fp = input_x
        for i in range(1, self.d):
            act_fp = self.act_fun_F[i](np.dot(self.weightsF[i - 1],self.add_bias(act_fp)))
        return act_fp

    def activation_fp_last_with_net(self, input_x):
        net = input_x
        act_fp = net
        for i in range(1, self.d):
            net = np.dot(self.weightsF[i - 1],self.add_bias(act_fp))
            act_fp = self.act_fun_F[i](net)
        return act_fp, net

    def activation_BP_last(self, input_y):
        act_BP = input_y
        for i in range(self.d - 1, 0, -1):
            act_BP = self.act_fun_F[i](np.dot(self.weightsB[i - 1],self.add_bias(act_BP)))
        return act_BP
