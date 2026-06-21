import numpy as np
import matplotlib.pyplot as plt
class RNN:
    def __init__(self, X, n_neurons,activation):
        self.activation=activation
        self.X = X
        self.T = max(self.X.shape[0]    )      # number of timesteps
        self.n_neurons = n_neurons
        
        # input dimension
        self.input_size = self.X.shape[1]

        self.Wx = 0.1 * np.random.randn(1,
                                        self.n_neurons)

        self.Wh = 0.1 * np.random.randn(self.n_neurons,
                                        self.n_neurons)

        self.Wy = 0.1 * np.random.randn(self.n_neurons,
                                        1)

        self.biases = 0.1 * np.random.randn(1,
                                            self.n_neurons)

        

    def forward(self):
        self.d_Wx = 0.1 * np.random.randn(1,
                                        self.n_neurons)

        self.d_Wh = 0.1 * np.random.randn(self.n_neurons,
                                        self.n_neurons)

        self.d_Wy = 0.1 * np.random.randn(self.n_neurons,
                                        1)

        self.d_biases = 0.1 * np.random.randn(1,
                                            self.n_neurons)
        
        self.H = [np.zeros((1, self.n_neurons))
                  for t in range(self.T + 1)]

        self.Y_hat     = np.zeros((1,self.T))
        
        ACT=[self.activation for t in range(self.T + 1)]
        ht=self.H[0]
        self.H,self.Y_hat,self.ACT=RNNcell(self,ACT,H,ht)
        
    def RNNcell(self,ACT,H,ht):
        for t, x in enumerate(X_t):
            x=x.reshape(1,1)
            out = np.dot(x,self.Wx) + np.dot(ht,self.Wh)\
                  + self.biases
                  
            ACT[t].forward(out)
            ht  = ACT[t].output

            y_hat_t  = np.dot(self.Wy, ht)
            
            H[t+1]   = ht
            Y_hat[t] = y_hat_t
            return (H,Y_hat,ACT)
    def backward(self,dvalues):
        T       = self.T
        H       = self.H
        X_t     = self.X_t
        
        ACT     = self.ACT
        
        dWx     = self.d_Wx
        dWy     = self.d_Wy
        dWh     = self.d_Wh
        Wy      = self.W_y
        Wh      = self.W_h
        
        dht     = np.dot(Wy.T,dvalues[-1].reshape(1,1))
        
        dbiases = self.d_biases
        for t in reversed(range(self.T)):
            dy = dvalues[t].reshape(1,1)
            xt = X_t[t].reshape(1,1)
            
            ACT[t].backward(dht)
            dtanh = ACT[t].dinputs
            
            dWx     += np.dot(dtanh,xt)
            dWy     += np.dot(H[t+1],dy).T
            dWh     += np.dot(H[t],dtanh.T)
            dbiases += dtanh
            
            dht = np.dot(Wh, dtanh) + np.dot(Wy.T,dy)

        self.dWx     = dWx
        self.dWy     = dWy
        self.dWh     = dWh
        self.dbiases = dbiases
        
        self.H       = H

        
        
            
        
    
class tanh:
    def forward(self,input_x):
        self.output=np.tanh(input_x)
    def backward(self,dvalues):
        self.d_input=np.dot(dvalues,1-self.output**2)
    
    
    
    
X_t = np.arange(-10,10,0.1)
X_t = X_t.reshape(len(X_t),1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t),1)
plt.plot(X_t, Y_t)
plt.show()

n_neurons = 500
rnn   = RNN(X_t, n_neurons)

Y_hat = rnn.Y_hat
H     = rnn.H
T     = rnn.T

ht = H[0]

for t, xt in enumerate(X_t):
    
    xt = xt.reshape(1,1)
    
    [ht, y_hat_t, out] = rnn.forward(xt, ht)
    
    H[t+1]   = ht
    print(Y_hat)
    Y_hat[0][t] = y_hat_t


dY         = Y_hat - Y_t
L          = 0.5*np.dot(dY.T,dY)/T

plt.plot(X_t, Y_t)
plt.plot(X_t, Y_hat[0])
plt.legend(['y', '$\hat{y}$'])
plt.show()

for h in H:
    plt.plot(np.arange(20), h[0:20], 'k-', linewidth = 1, alpha = 0.05)
