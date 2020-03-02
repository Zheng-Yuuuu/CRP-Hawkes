from crp_hawkes import crp_hawkes
from utilities import process_txt
import numpy as np 

# b ----- eta, zeta direction reversible


b_prior_mu = 0
b_prior_sigma = 5
zeta_prior_mu = 0
zeta_prior_sigma = 1
eta_prior_mu = 0
eta_prior_sigma = 1
max_b = 300
sample_num = 200

# d_num, u_id, observation_list, observation_table, ob_binary,binary_x, binary_y, T = process_txt('C:\\Users\\zy3\\work\\workspace\\fcp_windows\\radoslaw_email.txt')
# d_num, u_id, observation_list, observation_table, ob_binary,binary_x, binary_y, T = process_txt('out.radoslaw_email_email')
d_num, u_id, observation_list, observation_table, ob_binary,binary_x, binary_y, T = process_txt('syn_50')

print('overall T', T, 'd_num', d_num)
cluster_index = np.load('syn_c_index.npy')
# print(cluster_index)
fcp = crp_hawkes(b_prior_mu, b_prior_sigma, zeta_prior_mu, zeta_prior_sigma, 
eta_prior_mu, eta_prior_sigma, observation_list, max_b, c_max=200, 
d_num = d_num, u_id=u_id, nu=1, xi=1, b=10, zeta=1, eta=1, T=T)

over_likelog = np.zeros(sample_num)
scaling_likelog = np.zeros(sample_num)
for sample_index in range(sample_num):
    print(sample_index,'sample_index')
    
    print('sample z')
    fcp.sample_z()
    
    c_live = np.where(fcp.c_time[:,0]==0)[0]
    for c_send in c_live:
        for c_receive in c_live:
            for n in range(3):
                # print(c_send,c_receive,'sample hawkes')
                s = fcp.sample_hawkes(n,c_send, c_receive) 
                over_likelog[sample_index] = over_likelog[sample_index]+s
    
    for n in range(3):
        print('sample scaling')
        scaling_likelog[sample_index] = scaling_likelog[sample_index] + fcp.sample_scaling(n)

    print(fcp.b,'fcp.b')
    print(fcp.eta,'fcp.eat')
    print(fcp.zeta,'zeta')

    for m in c_live:
        for n in c_live:
            if fcp.Hawkes_b[m,n]>fcp.max or fcp.Hawkes_b[m,n]<fcp.min:
                print('b wrong')
            if fcp.Hawkes_eta[m,n]>fcp.max or fcp.Hawkes_eta[m,n]<fcp.min:
                print('eta wrong')
            if fcp.Hawkes_zeta[m,n]>fcp.max or fcp.Hawkes_zeta[m,n]<fcp.min:
                print('zeta wrong')
    
print(over_likelog)
print(scaling_likelog)


print(fcp.path_entity)

np.save('over_like',over_likelog)
np.save('scaling_like',scaling_likelog)

print(fcp.b,'fcp.b')
print(fcp.eta,'fcp.eat')
print(fcp.zeta,'zeta')
c_live = np.where(fcp.c_time[:,0]==0)[0]
for i in c_live:
    for j in c_live:
        print(fcp.Hawkes_b[i,j],i,j,'fcp.Hawkes_b[i,j],i,j')
        print(fcp.Hawkes_eta[i,j],i,j,'fcp.Hawkes_eta[i,j],i,j')
        print(fcp.Hawkes_zeta[i,j],i,j,'fcp.Hawkes_zeta[i,j],i,j')


    
    

