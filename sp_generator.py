import sympy as sp 
import numpy as np
import pickle
from tqdm import tqdm
class Quantum_walker(object):
    def __init__(self, position, n):
        self.n = n
        self.position = position

    def H(self, layer,bs,alpha):
        
        totH = sp.Matrix()
        for i in range(layer):
            single_Hgate = alpha[i]*sp.Matrix([[sp.cos(bs[i]), sp.I*sp.sin(bs[i])],[sp.I*sp.sin(bs[i]), sp.cos(bs[i])]])
            totH = sp.Matrix.diag(totH,single_Hgate)
        
        Hgate = sp.Matrix.diag(sp.Matrix.eye(self.n //2 - layer), totH, sp.Matrix.eye(self.n //2 - layer))
        
        self.position =  Hgate * self.position
    

    def P(self, layer, phi):
        totP = sp.Matrix()
        for i in range(layer):
            totP = sp.Matrix.diag(totP,1,sp.exp(sp.I*phi[i]))
            #totP = sp.Matrix.diag(totP,1,sp.exp(sp.I*phi[i]))
        
        Pgate = sp.Matrix.diag(sp.Matrix.eye(self.n //2 - layer), totP, sp.Matrix.eye(self.n //2 - layer))

        self.position = Pgate * self.position
    
    def simulate(self, total_layer, phi):
        index = 0
        for i in range(1, total_layer + 1):
            self.H(i, bs[index: index + i], alpha[index: index + i])
            self.P(i, phi[index: index + i])
            index += i
        self.H(total_layer + 1,bs[index: index + total_layer + 1], alpha[index: index + total_layer + 1])
def phi_num(layer):
    sum = 0
    for i in range(layer+1):
        sum += i
    return sum

layer = 11
position = np.zeros(2*layer+2)
position[layer] = 1
position = position.reshape(1,-1).T
walker = Quantum_walker(position, 2*layer + 2)
phi = sp.symbols(f'phi_1:{phi_num(layer) + 1}', real = True)
bs = sp.symbols(f'bs_1:{phi_num(layer+1) + 1}', real = True)
alpha = sp.symbols(f'alpha_1:{phi_num(layer+1) + 1}', real = True)
walker.simulate(layer,phi[0:phi_num(layer)+ 1])
path = f'11.pickle'

with open(path,'wb') as file:
    pickle.dump(walker.position.evalf(),file)