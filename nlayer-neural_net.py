import numpy as np
import tensorflow as tf

# Parameters
features = 3
hidden_layer = 3
outputs_num = 2
epochs = 2
batch_num = 10
sample_num = 100 # => m
alpha = 0.5  # Learning rate

rng = np.random.default_rng()
X = rng.integers(low=0, high=21, size=(features, sample_num))
Y = rng.integers(low=0, high=2, size=(outputs_num, sample_num))

class neural_net:
    activations = {'relu': tf.nn.relu,'tanh': tf.tanh, 'sigmoid': tf.sigmoid, 'softmax': tf.nn.softmax}
    
    def __init__(self, inputs :list, outputs :list, learning_rate :float, batche_num :int, num_layer :int, features :int, each_layer_neurons :list, each_layer_activation :list, sample_num :int, epoch :int, batch_num :int) -> None:
        self.activation_derivation = {'relu': self.relu_derivation,'tanh': self.tanh_derivation, 'sigmoid': self.sig_derivation, "softmax": self.sig_derivation}
        self.learning_rate = learning_rate
        self.numlayer = num_layer
        self.Y = outputs
        self.ac = each_layer_activation
        self.epochs = epoch
        self.n = sample_num

        self.batche = batche_num
        # Split data based on batches
        self.xbatches = np.split(inputs,batch_num,axis=1)
        self.ybatches = np.split(outputs,batch_num,axis=1)
        # Initializing weights and biases for each layer
        self.w = []
        self.b = []
        self.w.append(rng.integers(low=0, high=11, size=(each_layer_neurons[0], features)) * 0.01)
        for i in range(num_layer-1):
            self.w.append(rng.integers(low=0, high=11, size=(each_layer_neurons[i+1], each_layer_neurons[i])) * 0.01)

        for i in range(num_layer):
            self.b.append(np.zeros((each_layer_neurons[i],1)))
    
    def train(self):
        for i in range(self.epochs):
            loss = 0
            for c, b in enumerate(self.xbatches):
                self.a = []
                self.z = []
                self.a.append(b)
                self.forward()
                loss += self.loss(self.ybatches[c],self.a[self.numlayer])
                print(self.z)
                self.da = []
                self.dz = []
                self.dw = []
                self.db = []
                self.backward(c)
                self.update_parameters()
            print(f'Epoch {i + 1}, Loss(Cost): {loss/self.batche}')
    
    def forward(self):
        for i in range(self.numlayer):
            self.z.append(np.dot(self.w[i],self.a[i]) + self.b[i])
            self.a.append(np.array(self.activations[self.ac[i]](self.z[i])))

           
    def backward(self, batchnum):
        l = self.numlayer
        # self.da.append(-(np.divide(self.ybatches[batchnum], self.a[l]) - np.divide(1 - self.ybatches[batchnum], 1 - self.a[l])))
        # for i in range(l-1,-1,-1):
        #     derivative = self.activation_derivation[self.ac[i]](self.z[i])
        #     self.dz.append(np.multiply(self.da[l-1-i], derivative))
        #     self.dw.append(np.dot(self.dz[l-1-i], self.a[i].T) / (self.n // self.batche))
        #     self.db.append(np.sum(self.dz[l-1-i],axis=1,keepdims=True) / (self.n // self.batche))
        #     if i > 0:
        #         self.da.append(np.dot(self.w[i].T, self.dz[l-1-i]))
        self.dz.append(self.a[l]-self.ybatches[batchnum])
        self.dw.append(np.dot(self.dz[0], self.a[l-1].T) / (self.n // self.batche))
        self.db.append(np.sum(self.dz[0],axis=1,keepdims=True) / (self.n // self.batche))
        for i in range(l-2,-1,-1):
            derivative = self.activation_derivation[self.ac[i]](self.z[i])
            self.dz.append(np.multiply(np.dot(self.w[i+1].T, self.dz[l-i-2]), derivative))
            self.dw.append(np.dot(self.dz[l-1-i], self.a[i].T) / (self.n // self.batche))
            self.db.append(np.sum(self.dz[l-1-i],axis=1,keepdims=True) / (self.n // self.batche))

    def update_parameters(self):
        l = self.numlayer
        for i in range(l):
            self.w[i] -= self.learning_rate * self.dw[l-1-i]
            self.b[i] -= self.learning_rate * self.db[l-1-i]

    def loss(self, real, current):
        return -np.sum(real * np.log(current + 1e-8) + (1 - real) * np.log(1 - current + 1e-8)) / (self.n // self.batche)

    def tanh_derivation(self, mat):
        return (1 - np.tanh(mat) ** 2)
    
    def sig_derivation(self, mat):
        return mat * (1 - mat)
    
    def relu_derivation(self, mat):
        return (mat > 0).astype(float)
    
setup = neural_net(X,Y,alpha,batch_num,hidden_layer,features,[3,4,2],['relu','relu','softmax'],sample_num,epochs,batch_num)
setup.train()