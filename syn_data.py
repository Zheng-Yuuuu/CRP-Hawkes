import numpy as np 
import numpy.random as npr

def logit(x): # checked

    return 1/(1+np.exp(-x))

c_max = 2
d_num = 20
c_index = np.zeros(d_num,dtype=int)
c_index[int(d_num/2):] = 1
c_index = npr.permutation(c_index)

b = np.array([[100,1],[1,50]])
eta = np.array([[0.1,0.5],[0.5,0.1]])
zeta = np.array([[0.5,1],[1,0.5]])

x = []
for i in range(d_num):
    x.append([])
    for j in range(d_num):
        x[i].append([])

x_len = 0
for i in range(d_num):
    for j in range(i+1,d_num):

        s = 0
        T = 3
        while s<T:
            
            lambda_array = np.zeros(c_max)
            for m in range(c_max):
                mu = b[c_index[i],c_index[j]]
                eta_ij = eta[c_index[i],c_index[j]]
                zeta_ij = zeta[c_index[i],c_index[j]]
                phi = 0
                for t in x[j][i]:
                    phi = phi + eta_ij*np.exp(-zeta_ij*(s-t))
                lambda_array[m] = mu + phi

            lambda_ = np.sum(lambda_array)

            u = npr.uniform(low = 0, high = 1)
            w = -np.log(u)/lambda_
            s = s+w

            d = npr.uniform(low = 0, high = 1)

            lambda_array_ = np.zeros(c_max)
            for m in range(c_max):
                mu = b[c_index[i],c_index[j]]
                eta_ij = eta[c_index[i],c_index[j]]
                zeta_ij = zeta[c_index[i],c_index[j]]
                phi = 0
                for t in x[j][i]:
                    phi = phi + eta_ij*np.exp(-zeta_ij*(s-t))
                lambda_array_[m] = mu + phi

            lambda__ = np.sum(lambda_array_)
            
            if s<T:
                if d*lambda_<lambda__:
                    x_len = x_len + 1
                    if d*lambda_<lambda_array_[0]:
                        x[i][j].append(s)
                    else:
                        x[j][i].append(s)
                    

print(x[0][1])
print(x[1][0])


scale_T = 100000000

data = np.zeros((x_len,4))
print(data.shape)

index = 0
for i in range(d_num):
    for j in range(d_num):
        if i!=j:
            for t in x[i][j]:
                data[index,0] = i
                data[index,1] = j 
                data[index,3] = t * scale_T
                index = index + 1

np.savetxt('syn_50',data)
np.save('syn_c_index',c_index)