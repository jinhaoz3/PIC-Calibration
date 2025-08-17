import numpy as np
import pandas as pd

P_list = []
I_list = []
P = np.array(pd.read_excel('data.xlsx',sheet_name='Prob')).reshape(500,20)
I = np.array(pd.read_excel('data.xlsx',sheet_name='Current'))
for j in range(500):
    P_list.append(P[j]/np.sum(P[j]))
    I_list.append(I[j])

np.save('data_test/data_prob.npy',np.array(P_list))
np.save('data_test/data_voltage.npy',np.array(I_list))