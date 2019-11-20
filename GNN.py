import numpy as np
import copy

class GNN(object):
    def __init__(self, D = 8, T = 2, learning_rate = 0.001, epsilon = 0.001, 
                 epoch=10, batch_size=5, mu=0.9, random_state=0,
                 beta1=0.9, beta2=0.999, init_method=1, gd_method=0, verbose=0):
        
        np.random.seed(random_state)
            
        self.D = D
        self.T = T
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epoch = epoch
        self.batch_size = batch_size
        self.mu = mu
        self.random_state=random_state
        self.beta1 = beta1
        self.beta2 = beta2
        self.init_method = init_method
        self.gd_method = gd_method
        self.verbose = verbose
        self.t = 0
        
        self.W = np.random.normal(0, 0.4, (D, D))
        self.A = np.random.normal(0, 0.4, (1, D))
        self.b = 0
        
        self.W_momentum = np.zeros((D, D), float)
        self.A_momentum = np.zeros((1, D), float)
        self.b_momentum = 0
        
        self.W_v = np.zeros((D, D), float)
        self.A_v = np.zeros((1, D), float)
        self.b_v = 0
    
    def ReLU(self, x):
        for i in range(self.D):
            if x[i,0] < 0:
                x[i,0] = 0
        return x
    
    def sigmoid(self, s):
        if s > 20:
            return 1 / (1 + np.exp(-20))
        else:
            return 1 / (1 + np.exp(-s))
    
    def calc_loss(self, y, s):
        return -y * np.log(self.sigmoid(s)) - (1 - y) * np.log(1 - self.sigmoid(s))
    
    #特徴ベクトルの初期化
    def init_x(self, X):
        num_samples = X.shape[0]
        x = np.zeros((num_samples, self.D), float)
        
        for i in range(num_samples):
            if self.init_x == 0:
                x[i,0] = 1
            elif self.init_method == 1:
                for j in range(self.D):
                    x[i, j] = np.random.rand()
            
        return x
    
    #集約後のxとaを返す関数
    def update_x(self, adj_matrix, X):
        num_samples = adj_matrix.shape[0]
        
        XX = np.empty((num_samples, self.D), float)
        a = np.empty((num_samples, self.D), float)
        
        for i in range(num_samples):
            for j in range(self.D):
                a[i,j] = 0
            
            for j in range(num_samples):
                if(adj_matrix[i, j] == 1):
                    for k in range(self.D):
                        a[i,k] += X[j,k]
            
            XX[i, :] = self.ReLU(np.dot(self.W, a[i,:].reshape(-1,1))).reshape(1,-1)
            
        return XX, a
    
    def gradient_descent(self, adj_matrix, y, a, cur_loss, h):
        
        num_samples = adj_matrix.shape[0]
        W_update = np.empty((self.D, self.D), float)
        
        #W
        W_d = copy.copy(self.W)
        for i in range(self.D):
            for j in range(self.D):
                W_d[i, j] += self.epsilon
                
                X = np.empty((num_samples, self.D), float)
                for k in range(num_samples):
                    X[k,:] = self.ReLU(np.dot(W_d, a[k, :].reshape(-1, 1))).reshape(1, -1)
                
                s, h = self.calc_s(X)
                W_update[i, j] = (self.calc_loss(y, s) - cur_loss) / self.epsilon
                W_d[i, j] -= self.epsilon
                
        #A    
        A_update = np.empty((1, self.D), float)
        A_d = copy.copy(self.A)
        for i in range(self.D):
            
            A_d[0,i] += self.epsilon
            
            s = np.dot(A_d, h)[0,0] + self.b
            A_update[0,i] = (self.calc_loss(y, s) - cur_loss) / self.epsilon
            
            A_d[0,i] -= self.epsilon
        
        #b
        s = np.dot(self.A, h)[0, 0] + self.b + self.epsilon
        b_update = (self.calc_loss(y, s) - cur_loss)  / self.epsilon
    
        return W_update, A_update, b_update

    #パラメータを初期化する
    def init_parameter(self):
    
        np.random.seed(self.random_state)
        self.t = 0
        
        self.W = np.random.normal(0, 0.4, (D, D))
        self.A = np.random.normal(0, 0.4, (1, D))
        self.b = 0
        
        self.W_momentum = np.zeros((D, D), float)
        self.A_momentum = np.zeros((1, D), float)
        self.b_momentum = 0
        
        self.W_v = np.zeros((D, D), float)
        self.A_v = np.zeros((1, D), float)
        self.b_v = 0

        
    def fit(self, X = None , y = None, X_test = None, y_test = None):
        
        num_samples = X.shape[0]
        
        for i in range(self.epoch):
            batch_mask = np.empty(num_samples, bool)
            batch_idx = np.empty(num_samples, int)
            for j in range(num_samples):
                batch_mask[j] = True
                batch_idx[j] = j
                
            used_num = 0
            
            W_update = np.zeros((self.D, self.D), float)
            A_update = np.zeros((1, self.D), float)
            b_update = 0
            
            loss = 0
            
            
            while True:
                self.t += 1
                #バッチサイズが残りのデータ数以下の時
                if used_num + self.batch_size >= num_samples:
                    idx = batch_idx[batch_mask]
                    for k in idx:
                        W_tmp, A_tmp, b_tmp, loss_tmp = self.fit_one(X[k], y[k])
                        W_update += W_tmp
                        A_update += A_tmp
                        b_update += b_tmp
                        loss += loss_tmp
                        
                    W_update /= num_samples - used_num
                    A_update /= num_samples - used_num
                    b_update /= num_samples - used_num
                    
                    self.update_params(W_update, A_update, b_update)
                    
                    break
                #バッチサイズが残りのデータ数以上の時
                else:
                    idx = np.random.choice(batch_idx[batch_mask], size=self.batch_size, replace=False)
                    
                    batch_mask[idx] = False
                    for k in idx:
                        W_tmp, A_tmp, b_tmp, loss_tmp = self.fit_one(X[k], y[k])
                        W_update += W_tmp
                        A_update += A_tmp
                        b_update += b_tmp
                        loss += loss_tmp
                        
                    used_num += self.batch_size
                    
                    W_update /= self.batch_size
                    A_update /= self.batch_size
                    b_update /= self.batch_size
                    
                    self.update_params(W_update, A_update, b_update)

            if self.verbose > 0:
                print("epoch: %d Train Loss %f" % (i+1, loss / num_samples))
                if self.verbose >= 2:
                    pred = self.predict(X_test)
                    print("epoch: %d Test accuracy %f" % (i+1, self.accuracy_score(pred, y_test)))
        
    def update_params(self, W_update, A_update, b_update):
        
        #SGD
        if self.gd_method == 0:
            self.W = self.W - self.learning_rate * W_update
            self.A = self.A - self.learning_rate * A_update
            self.b = self.b - self.learning_rate * b_update
            
        #Momentum SGD
        elif self.gd_method == 1:
            self.W = self.W - self.learning_rate * W_update + self.mu * self.W_momentum
            self.A = self.A - self.learning_rate * A_update + self.mu * self.A_momentum
            self.b = self.b - self.learning_rate * b_update + self.mu * self.b_momentum

            self.W_momentum = - self.learning_rate * W_update + self.mu * self.W_momentum
            self.A_momentum = - self.learning_rate * A_update + self.mu * self.A_momentum
            self.b_momentum = - self.learning_rate * b_update + self.mu * self.b_momentum
            
        #Adam
        else:
            self.W_momentum = self.beta1 * self.W_momentum + (1 - self.beta1) * W_update
            self.A_momentum = self.beta1 * self.A_momentum + (1 - self.beta1) * A_update
            self.b_momentum = self.beta1 * self.b_momentum + (1 - self.beta1) * b_update

            self.W_v = self.beta2 * self.W_v + (1 - self.beta2) * np.square(W_update)
            self.A_v = self.beta2 * self.A_v + (1 - self.beta2) * np.square(A_update)
            self.b_v = self.beta2 * self.b_v + (1 - self.beta2) * np.square(b_update)

            self.W = self.W - self.learning_rate * (self.W_momentum / (1 - self.beta1 ** self.t)) / (np.sqrt(self.W_v / (1 - self.beta2 ** self.t)) + 1e-8)
            self.A = self.A - self.learning_rate * (self.A_momentum / (1 - self.beta1 ** self.t)) / (np.sqrt(self.A_v / (1 - self.beta2 ** self.t)) + 1e-8)
            self.b = self.b - self.learning_rate * (self.b_momentum / (1 - self.beta1 ** self.t)) / (np.sqrt(self.b_v / (1 - self.beta2 ** self.t)) + 1e-8)

    def accuracy_score(self, pred, y):
        
        correct = 0.0
        for i in range(y.shape[0]):
            if pred[i] == y[i]:
                correct += 1
                
        return correct / y.shape[0]
    
    #一つのグラフデータを学習する関数
    def fit_one(self, adj_matrix, y):
        
        num_samples = adj_matrix.shape[0]
        
        #xを初期化
        x = self.init_x(adj_matrix)
        
        #集約
        for i in range(self.T):
            x, a = self.update_x(adj_matrix, x)
     
        #READOUT
        s, h = self.calc_s(x)
        cur_loss = self.calc_loss(y, s)
        
        W_update, A_update, b_update = self.gradient_descent(adj_matrix, y, a, cur_loss, h)
        
        return W_update, A_update, b_update, cur_loss
        
    #sを計算し，返り値としてsとhを返す
    def calc_s(self, x):
        num_samples = x.shape[0]
        h = np.zeros((self.D, 1), float)
        
        #READOUT
        for i in range(num_samples):
            for j in range(self.D):
                h[j, 0] += x[i, j]
                
        #calc s
        s = np.dot(self.A, h)[0,0] + self.b
        
        return s, h
    
    def predict(self, X):
        num_samples = X.shape[0]
        
        pred = np.zeros(num_samples, int)
        
        for i in range(num_samples):
            s = self.predict_one(X[i])
            
            p = self.sigmoid(s)
            
            if p >= 0.5:
                pred[i] = 1
                
        return pred
            
    def predict_one(self, adj_matrix):
        num_samples = adj_matrix.shape[0]
        
        #xを初期化
        x = self.init_x(adj_matrix)
        
        #集約
        for i in range(self.T):
            x, a = self.update_x(adj_matrix, x)
    
        #READOUT
        s, h = self.calc_s(x)  

        return s
    
    #学習したパラメータ(W, A, b)を出力する
    def print_parameter(self):
        print("W %d %d" % (self.D, self.D))
        print(self.W)
        print()
        
        print("A %d" % (self.D))
        print(self.A)
        print()
        
        print("b")
        print(self.b)
        
