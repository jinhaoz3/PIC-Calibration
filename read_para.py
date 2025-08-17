import torch
import numpy as np


params_path = f'11layer(1225).pth'
state_dict = torch.load(params_path)

np.savetxt('para_a.csv',state_dict['0.a'],delimiter=',')
np.savetxt('para_b.csv',state_dict['0.b'],delimiter=',')
np.savetxt('para_bs.csv',state_dict['1.bs'],delimiter=',')
np.savetxt('para_coupling.csv',state_dict['2.coupling'],delimiter=',')
print(state_dict)