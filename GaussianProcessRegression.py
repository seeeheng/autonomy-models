import numpy as np
import matplotlib.pyplot as plt
import utils

class GaussianProcessRegression:
    LR = 0.01
    HP_ITERATIONS = 5000
    def __init__(self, hp_l, hp_ampl, hp_n, x_min, x_max, x_resolution):
        self.hp_l = hp_l
        self.hp_ampl = hp_ampl
        self.hp_n = hp_n

        self.x_range = np.linspace(x_min, x_max, num=x_resolution)
        self.K = None
        self.y_mu = None
        self.y_std = None

    def generate_K_matrix(self, x_train):
        """ Creates K matrix for all x_train possibilities

        Args:
            x_train = training data to train on

        """
        # Creating empty matrix of dimensions: [len(x_train), len(x_train)]
        K = np.zeros([len(x_train), len(x_train)])
        
        # Populating all columns to get K matrix of all possible X combinations.
        for i in range(len(x_train)):
            for j in range(len(x_train)):
                K[i,j] = utils.k_squared_exp(x_train[i], x_train[j], hp_ampl=self.hp_ampl**2, hp_l=self.hp_l)
                if i==j:
                    # A little noise goes a long way.
                    K[i,j] += self.hp_n**2
        self.K = K
        return K

    def gaussian_process_regression(self, x_train, y_train):
        """ Regression, using bayes rule.

        Args:
            x_train: training data
            y_train: training labels
        """
        print("Regression using hp_l={}, hp_ampl={}, hp_n={}".format(self.hp_l, self.hp_ampl, self.hp_n))

        y_mu = np.zeros(len(self.x_range))
        y_std = np.zeros(len(self.x_range))
        K_inv = np.linalg.inv(self.K)

        k_vec = np.zeros([1, len(x_train)])

        for i in range(len(self.x_range)):
            for j in range(len(x_train)):
                k_vec[0,j] = utils.k_squared_exp(self.x_range[i], x_train[j], hp_ampl=self.hp_ampl**2, hp_l=self.hp_l)
            y_mu[i] = np.matmul(np.matmul(k_vec, K_inv), y_train.transpose())
            y_std[i] = np.sqrt(self.hp_n**2 + self.hp_ampl**2 - np.matmul(np.matmul(k_vec,K_inv),k_vec.transpose()))

        self.y_mu = y_mu
        self.y_std = y_std
        return y_mu, y_std    

    def gaussian_learn_hyperparameters(self, x_train, y_train):
        """ Learn hyperparameters using gradient asc

        Args:
            x_train: training data
            y_train: training labels
        """
        hp_l = self.hp_l
        hp_ampl = self.hp_ampl
        hp_n = self.hp_n
        K_shape = [len(x_train),len(x_train)] 
        for i_learn in range(self.HP_ITERATIONS):
            # Initializing matrices
            K = np.zeros(K_shape)
            dK_l = np.zeros(K_shape)
            dK_a = np.zeros(K_shape)
            dK_n = np.zeros(K_shape)
            
            # Calculating gradients
            for i in range(K_shape[0]):
                for j in range(K_shape[1]):
                    K[i,j] = utils.k_squared_exp(x_train[i], x_train[j], hp_ampl=hp_ampl**2, hp_l=hp_l)
                    dK_l[i,j] = utils.dk_squared_exp_l(x_train[i], x_train[j], hp_ampl=hp_ampl**2, hp_l=hp_l)
                    dK_a[i,j] = utils.dk_squared_exp_a(x_train[i], x_train[j], hp_ampl=hp_ampl**2, hp_l=hp_l)
                    dK_n[i,j] = 0.
                    if i==j:
                        K[i,j] += hp_n**2
                        dK_n[i,j] = 2*hp_n
            K_inv = np.linalg.inv(K)
            alpha = np.matmul(K_inv, y_train.transpose())
            aaT_K_inv = np.outer(alpha, alpha.transpose())-K_inv

            # Learning
            hp_l += self.LR * (0.5 * np.trace(np.matmul(aaT_K_inv, dK_l)))
            hp_ampl += self.LR * (0.5 * np.trace(np.matmul(aaT_K_inv, dK_a)))
            hp_n += self.LR * (0.5 * np.trace(np.matmul(aaT_K_inv, dK_n)))
        self.hp_l = hp_l
        self.hp_ampl = hp_ampl
        self.hp_n = hp_n
        print("Hyperparmeters learnt:\nLength={}, Amplitude={}, Noise={}".format(hp_l, hp_ampl, hp_n))
        print("Generating K Matrix based on new Hyperparmeters...")
        self.generate_K_matrix(x_train)

    def predict(self, x):
        """ Returns predictions based on currently trained Gaussian Process

        Args:
            x: accepts both singular number / np array.
        """
        if type(self.y_mu) != np.ndarray:
            print("Gaussian Process Regression hasn't been called.")
            return 1
        else:
            index_bin = np.digitize(x, self.x_range)
            try:
                assert len(index_bin[index_bin >= len(self.x_range)])<1
                # print(self.x_range)
            except AssertionError:
                print("ERROR: Were you trying to predict a value greater / lower than your current x_max / x_min?".format(self.x_range[-1]))
                return
            return np.random.normal(loc=self.y_mu[index_bin],scale=self.y_std[index_bin])          

def main():
    x_train = np.array([0, -7.2, 1.1, 5.6, 6.1])
    y_train = np.array([2.5, 1.1, 3.1, 1.2, 1.31])

    l = 1.5 # length
    a = 1 # amplitude (sigma_a ** 2)
    n = 0.05**2

    gaussian_process = GaussianProcessRegression(l, a, n, -10, 10, 1000)
    # gaussian_process.generate_K_matrix(x_train)
    gaussian_process.gaussian_learn_hyperparameters(x_train, y_train)
    gaussian_process.gaussian_process_regression(x_train, y_train)
    print("Prediction for x=[5,2,9]: {}".format(gaussian_process.predict(np.array([5,2,9]))))
    print("Prediction for x=2: {}".format(gaussian_process.predict(2)))
    
if __name__ == "__main__":
    main()
