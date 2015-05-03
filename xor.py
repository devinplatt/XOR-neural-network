# A simple neural network which learns XOR using back propagation.

import random
from math import exp

# See wikipedia for the neural network structure
# http://en.wikipedia.org/wiki/Feedforward_neural_network
# http://en.wikipedia.org/wiki/File:XOR_perceptron_net.png
# Pitiful inline diagram:
#           z-b3                Output
#       /   |     \
#     w4    w5     w6
#    /      |       \
#  h1-b0   h2-b1     h3-b2      Hidden Units
#  /     /   \       /
# w0    w1    w2    w3
#  \   /       \   /     
#    x0        x1               Input

# activation function: logistic
def g(z):
  return 1/(1 + exp(-z))

def g_prime(z):
  return g(z) * g(-z)

# Training / Testing data
# (Train, test are same since we are modeling a known boolean function)
ex1 = [0,0,0]
ex2 = [0,1,1]
ex3 = [1,0,1]
ex4 = [1,1,0]
ex = [ex1, ex2, ex3, ex4]

class net:

  def __init__(self):
    self.eta = .5
    self.w = [random.random() for _ in range(7)]
    self.b = [random.random() for _ in range(4)]
    self.delta = [0, 0, 0, 0]
    self.net = [0, 0, 0, 0]

  # Working weights according the the reference diagram from Wikipedia.
  # This allows us to test the network structure and classification 
  # independently of training.
  def set_weights(self):
    self.w = [1,1,1,1,1,-2,1]
    self.b = [-1,-2,-1,0]

  def classify(self, x, p = False):
    self.net[0] = self.w[0] * x[0] + self.b[0] * 1
    self.net[1] = self.w[1] * x[0] + self.w[2] * x[1] + self.b[1] * 1
    self.net[2] = self.w[3] * x[1] + self.b[2] * 1
    self.net[3] = (self.w[4] * g(self.net[0])
                  + self.w[5] * g(self.net[1])
                  + self.w[6] * g(self.net[2])
                  + self.b[3] * 1)
    if p:
      print(g(self.net[3]))
    if g(self.net[3]) > .5:
      return 1
    else:
      return 0

  def train(self, x):
    y = self.classify(x)
    t = x[2]

    # Back propagation of error
    # Note that we calculate ALL deltas before updating any weights
    # (It's hard for me to find a webpage that explains why this
    # order of operations is important)
    self.delta[3] = (t-y) * g_prime(self.net[3])

    self.delta[0] = (self.delta[3] * self.w[4]) * g_prime(self.net[0])
    self.delta[1] = (self.delta[3] * self.w[5]) * g_prime(self.net[1])
    self.delta[2] = (self.delta[3] * self.w[6]) * g_prime(self.net[2])

    # Weight update
    self.w[4] = self.w[4] + self.eta * self.delta[3] * g(self.net[0])
    self.w[5] = self.w[5] + self.eta * self.delta[3] * g(self.net[1])
    self.w[6] = self.w[6] + self.eta * self.delta[3] * g(self.net[2])
    self.b[3] = self.b[3] + self.eta * self.delta[3] * 1

    self.w[0] = self.w[0] + self.eta * self.delta[0] * x[0]
    self.b[0] = self.b[0] + self.eta * self.delta[0] * 1
    self.w[1] = self.w[1] + self.eta * self.delta[1] * x[0]
    self.w[2] = self.w[2] + self.eta * self.delta[1] * x[1]
    self.b[1] = self.b[1] + self.eta * self.delta[1] * 1
    self.w[3] = self.w[3] + self.eta * self.delta[2] * x[1]
    self.b[2] = self.b[2] + self.eta * self.delta[2] * 1

  def details(self):
    print('output activations')
    for x in ex:
      nn.classify(x, True)
    print("classifications")
    for x in ex:
      print('{0}: {1}'.format(x[0:2], nn.classify(x)))
    print('biases')
    print(self.b)
    print('weights')
    print(self.w)

nn = net()

m = 30
t = [0 for _ in range(m)]
y = [0 for _ in range(m)]
for i in range(m):
  x = random.choice(ex)
  t[i] = x[2]
  y[i] = nn.classify(x)

max_iter = 50000
accuracy = sum(t[i] == y[i] for i in range(m)) / float(m)
n = 0
while accuracy < .99 and n < max_iter:
  x = random.choice(ex)
  guess = nn.classify(x)
  i = n % m
  t[i] = x[2]
  y[i] = guess
  accuracy = sum(t[i] == y[i] for i in range(m)) / float(m)
  n = n+1
  nn.train(x)

accuracy = sum(t[i] == y[i] for i in range(m)) / float(m)
print('number trials: {0}'.format(n))
print('accuracy: {0}'.format(accuracy))

nn.details()
