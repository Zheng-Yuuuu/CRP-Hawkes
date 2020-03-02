import numpy as np 
import copy
import numpy.random as npr

npr.seed(0)

class crp_hawkes(object):

    def __init__(self, b_prior_mu, b_prior_sigma, zeta_prior_mu, zeta_prior_sigma, eta_prior_mu, eta_prior_sigma,
                 observation_list, max_b, particle_num = 100, c_max=20, 
                 d_num = 0, u_id = [], nu = 1, xi = 1, b = 1, zeta = 1, eta = 1, T = 0):
                
        # auxilary
        
        self.u_id = u_id
        self.undefine_para_value = -100
        self.c_max = c_max

        self.min = -10
        self.max = 10

        #hyperparamter
        
        self.b_prior_mu = b_prior_mu
        self.b_prior_sigma = b_prior_sigma
        self.zeta_prior_mu = zeta_prior_mu
        self.zeta_prior_sigma = zeta_prior_sigma
        self.eta_prior_mu = eta_prior_mu
        self.eta_prior_sigma = eta_prior_sigma
        self.b_max = max_b
        
        self.T = T
        
        # parameter for chinese restaurant process
        self.nu = nu # fragmentation parameter
        self.xi = xi # coagulation parameter
        
        # hawkes scaling parameter

        self.b = b  
        self.eta = eta    
        self.zeta = zeta
    
        # hawkes parameter for community pair(b,zeta,eta)
        
        self.Hawkes_b = np.zeros((self.c_max,self.c_max))+self.undefine_para_value
        self.Hawkes_eta = np.zeros((self.c_max,self.c_max))+self.undefine_para_value
        self.Hawkes_zeta = np.zeros((self.c_max,self.c_max))+self.undefine_para_value
        
        
        # FCPNode information
        
        self.entity_num = np.zeros(c_max,dtype=int)
        self.entity_list = [[] for i in range(c_max)]
        
        self.c_time = np.zeros((self.c_max,2)) # record beginning and ending time
        self.c_time[:,0] = -1 # use c_time[:,0] to check the community is alive
        self.c_time[:,1] = self.T # set initializtion of ending time being self.T for convenience
        
        
        # entity information
        
        self.d_num = d_num # entity number
        self.observation_list = observation_list # observation_list
        self.path_entity = [[] for i in range(self.d_num)] # community path for each entity
        
        
        #initialization
        
        # initialize c_time (community at time 0)

        initial_c_num = 1

        for i in range(self.d_num):
            c_index = npr.randint(initial_c_num, size = 1)[0]
            self.path_entity[i]=c_index
            if c_index == initial_c_num-1:
                self.c_time[c_index,0] = 0
                initial_c_num = initial_c_num+1
        
        # initialize  ----------entity_list,entity_num

        for i in range(self.d_num):
            
            c_index = self.path_entity[i]
            self.entity_list[c_index].append(i)
            self.entity_num[c_index] = self.entity_num[c_index]+1

        # initialize ----------- hawkes parameter

        index = np.where(self.c_time[:,0]==0)[0]

        for p in index:
            for q in index:
                v = npr.normal(loc = self.b_prior_mu, scale = self.b_prior_sigma)
                self.Hawkes_b[p,q] = np.clip(v, self.min, self.max)
                v = npr.normal(loc = self.eta_prior_mu, scale = self.eta_prior_sigma)
                self.Hawkes_eta[q,p] = np.clip(v, self.min, self.max)
                v = npr.normal(loc = self.zeta_prior_mu, scale = self.zeta_prior_sigma)
                self.Hawkes_zeta[q,p] = np.clip(v, self.min, self.max)


    def preprocess_i(self, i): # 

        c = self.path_entity[i]
        # print(c,'c in preprocess')
        
        self.entity_list[c].remove(i)
        self.entity_num[c] = self.entity_num[c]-1

        if self.entity_num[c] == 0:
            self.c_time[c,0] = -1
            self.remove_hawkes_parameter(c)


    def remove_hawkes_parameter(self,community_index):
        
        self.Hawkes_b[community_index,:] = self.undefine_para_value 
        self.Hawkes_b[:,community_index] = self.undefine_para_value
        self.Hawkes_eta[community_index,:] = self.undefine_para_value
        self.Hawkes_eta[:,community_index] = self.undefine_para_value
        self.Hawkes_zeta[community_index,:] = self.undefine_para_value
        self.Hawkes_zeta[:,community_index] = self.undefine_para_value


    def sample_z(self):

        for i in range(self.d_num):
            

            # print(i, 'entity index in sample z')
            self.preprocess_i(i)

            c_live = np.where(self.c_time[:,0]==0)[0]
            prior = np.zeros(len(c_live)+1)
            for c in range(len(c_live)):
                prior[c] =  self.entity_num[c_live[c]]/(self.d_num-1+self.nu/self.xi)
            
            prior[-1] = self.nu/self.xi/(self.d_num-1+self.nu/self.xi)

            v = npr.normal(loc = self.b_prior_mu, scale = self.b_prior_sigma,size = len(c_live))
            new_b = np.clip(v, self.min, self.max)
            v = npr.normal(loc = self.eta_prior_mu, scale = self.eta_prior_sigma, size = len(c_live))
            new_eta = np.clip(v, self.min, self.max)
            v = npr.normal(loc = self.zeta_prior_mu, scale = self.zeta_prior_sigma, size = len(c_live))
            new_zeta = np.clip(v, self.min, self.max)

            v = npr.normal(loc = self.b_prior_mu, scale = self.b_prior_sigma, size = len(c_live))
            new_b_ = np.clip(v, self.min, self.max)
            v = npr.normal(loc = self.eta_prior_mu, scale = self.eta_prior_sigma, size = len(c_live))
            new_eta_ = np.clip(v, self.min, self.max)
            v = npr.normal(loc = self.zeta_prior_mu, scale = self.zeta_prior_sigma, size = len(c_live))
            new_zeta_ = np.clip(v, self.min, self.max)

            like = np.zeros(len(c_live)+1)

            for n in range(len(c_live)):

                for j in range(self.d_num):
                    if i!=j:

                        x_ij = self.observation_list[i][j]
                        x_ji = self.observation_list[j][i]

                        b = self.Hawkes_b[c_live[n],self.path_entity[j]]
                        eta = self.Hawkes_eta[c_live[n],self.path_entity[j]]
                        zeta = self.Hawkes_zeta[c_live[n],self.path_entity[j]]

                        b_ = self.b * logit(b)
                        eta_ = self.eta * logit(eta)
                        zeta_ = self.zeta * logit(zeta)

                        # print('1 sample z')
                        like[n] = like[n] + self.cal_hawkes_likelihood_(b_, eta_, zeta_, x_ij, x_ji)

                        b = self.Hawkes_b[self.path_entity[j],c_live[n]]
                        eta = self.Hawkes_eta[self.path_entity[j],c_live[n]]
                        zeta = self.Hawkes_zeta[self.path_entity[j],c_live[n]]

                        b_ = self.b * logit(b)
                        eta_ = self.eta * logit(eta)
                        zeta_ = self.zeta * logit(zeta)
                        # print('2 sample z')
                        like[n] = like[n] + self.cal_hawkes_likelihood_(b_, eta_, zeta_, x_ji, x_ij)

            for j in range(self.d_num):
                if i!=j:

                    x_ij = self.observation_list[i][j]
                    x_ji = self.observation_list[j][i]

                    cluster_j = self.path_entity[j]
                    cluster_index = np.where(c_live == cluster_j)[0]

                    j_b = new_b[cluster_index]
                    j_eta = new_eta[cluster_index]
                    j_zeta = new_zeta[cluster_index]

                    j_b_ = new_b_[cluster_index]
                    j_eta_ = new_eta_[cluster_index]
                    j_zeta_ = new_zeta_[cluster_index]

                    b_ = self.b * logit(j_b)
                    eta_ = self.eta * logit(j_eta)
                    zeta_ = self.zeta * logit(j_zeta)
                    
                    # print('3 sample z')
                    like[-1] = like[-1] + self.cal_hawkes_likelihood_(b_, eta_, zeta_, x_ij, x_ji)

                    b_ = self.b * logit(j_b_)
                    eta_ = self.eta * logit(j_eta_)
                    zeta_ = self.zeta * logit(j_zeta_)

                    # print('4 sample z')
                    like[-1] = like[-1] + self.cal_hawkes_likelihood_(b_, eta_, zeta_, x_ji, x_ij)
            
            post = np.log(prior)+like

            normalize_weight(post)

            index = np.argmax(npr.multinomial(1, post))

            if index != len(post)-1:

                c = c_live[index]
                self.path_entity[i] = c
                self.entity_list[c].append(i)
                self.entity_num[c] = self.entity_num[c]+1
            else:

                c_null = np.where(self.c_time[:,0]<0)[0]
                if c_null != []:
                    c_new = c_null[0]
                    self.path_entity[i] = c_new 
                    self.entity_list[c_new].append(i)
                    self.entity_num[c_new] = self.entity_num[c_new]+1
                    self.c_time[c_new,0] = 0

                    self.Hawkes_b[c_new,c_live] = new_b[:]
                    self.Hawkes_eta[c_new,c_live] = new_eta[:]
                    self.Hawkes_zeta[c_new,c_live] = new_zeta[:]

                    self.Hawkes_b[c_live,c_new] = new_b_[:]
                    self.Hawkes_eta[c_live,c_new] = new_eta_[:]
                    self.Hawkes_zeta[c_live,c_new] = new_zeta_[:]

                    v = npr.normal(loc = self.b_prior_mu, scale = self.b_prior_sigma, size = 1)
                    new_b = np.clip(v, self.min, self.max)
                    v = npr.normal(loc = self.eta_prior_mu, scale = self.eta_prior_sigma, size = 1)
                    new_eta = np.clip(v, self.min, self.max)
                    v = npr.normal(loc = self.zeta_prior_mu, scale = self.zeta_prior_sigma, size = 1)
                    new_zeta = np.clip(v, self.min, self.max)

                    self.Hawkes_b[c_new,c_new] = new_b
                    self.Hawkes_eta[c_new,c_new] = new_eta
                    self.Hawkes_zeta[c_new,c_new] = new_zeta
 
                else:
                    
                    print('wrong!!!')
                    exit()

                

    def cal_hawkes_likelihood_(self, b, eta, zeta, t_base, t_trigger):
        
        # calculate the likelihood
        # print(b, eta, zeta, t_base, t_trigger)
        s = 0
        for i in range(len(t_base)):
            s_ = b
            for j in range(len(t_trigger)):
                if t_base[i]>t_trigger[j]:
                    # print(eta[j]*np.exp(-zeta[j]*(t_base[i]-t_trigger[j])),'eta[j]*np.exp(-zeta[j]*(t_base[i]-t_trigger[j]))')
                    s_ = s_ + eta*np.exp(-zeta*(t_base[i]-t_trigger[j]))
            s = s + np.log(s_)

        s = s - b*self.T
        for j in range(len(t_trigger)): 
            # print(- eta[j]*(1 - np.exp( -zeta[j]*(self.T-t_trigger[j])))/zeta[j],'- eta[j]*(1 - np.exp( -zeta[j]*(self.T-t_trigger[j])))/zeta[j]')
            s = s - eta*(1 - np.exp( -zeta*(self.T-t_trigger[j])))/zeta
        
        # print((s),'s in cal')
        return s 


    def sample_scaling(self, para_select):

        # para_select  0:b       1:eta        2:zeta

        loglike = 0
        loglike_ = 0     

        new_b = 0
        new_eta = 0
        new_zeta = 0

        if para_select == 0:
            new_b = npr.uniform(low = 0, high = self.b_max)
        if para_select == 1:
            new_eta = npr.uniform(low = 0, high = 1/self.zeta)
        if para_select == 2:
            new_zeta = npr.uniform(low = 0, high = 1/self.eta)

        for i in range(self.d_num):
            for j in range(self.d_num):
                if i!=j:
                    
                    x_ij = self.observation_list[i][j]
                    x_ji = self.observation_list[j][i]

                    b = self.Hawkes_b[self.path_entity[i],self.path_entity[j]]
                    eta = self.Hawkes_eta[self.path_entity[i],self.path_entity[j]]
                    zeta = self.Hawkes_zeta[self.path_entity[i],self.path_entity[j]]

                    b_ = self.b * logit(b)
                    eta_ = self.eta * logit(eta)
                    zeta_ = self.zeta * logit(zeta)

                    loglike = loglike + self.cal_hawkes_likelihood_(b_, eta_, zeta_, x_ij, x_ji)
                    
                    if para_select == 0:
                        b_ = new_b * logit(b)
                    if para_select == 1:
                        eta_ = new_eta * logit(eta)
                    if para_select == 2:
                        zeta_ = new_zeta * logit(zeta)

                    loglike_ = loglike_ + self.cal_hawkes_likelihood_(b_, eta_, zeta_, x_ij, x_ji)

        u = loglike_ - loglike
        if u>0:
            if para_select == 0:
                self.b = new_b
            if para_select == 1:
                self.eta = new_eta
            if para_select == 2:
                self.zeta = new_zeta
            return loglike_
        else:
            u = np.exp(u)
            if npr.uniform(low=0.0, high=1.0) < u:
                if para_select == 0:
                    self.b = new_b
                if para_select == 1:
                    self.eta = new_eta
                if para_select == 2:
                    self.zeta = new_zeta
                return loglike_
            else:
                return loglike


    def sample_hawkes(self, para_select, c_send, c_receive):
        
        # not consider bound yet!!!
        
        # detemine the community location
        new_b = 0
        new_eta = 0
        new_zeta = 0

        mu = 0
        sigma = 1

        if para_select == 0:
            new_b = npr.normal(self.Hawkes_b[c_send,c_receive], 1)
            mu = self.Hawkes_b[c_send,c_receive]
        if para_select == 1:
            new_eta = npr.normal(self.Hawkes_eta[c_send,c_receive], 1)
            mu = self.Hawkes_eta[c_send,c_receive]
        if para_select == 2:
            new_zeta = npr.normal(self.Hawkes_zeta[c_send,c_receive], 1)
            mu = self.Hawkes_zeta[c_send,c_receive]
        
        new_b = np.clip(new_b,self.min,self.max)
        new_eta = np.clip(new_eta,self.min,self.max)
        new_zeta = np.clip(new_zeta,self.min,self.max)

        loglike = 0
        loglike_ = 0

        for i in self.entity_list[c_send]:
            for j in self.entity_list[c_receive]:

                if i!=j:
        
                    x_ij = self.observation_list[i][j]
                    x_ji = self.observation_list[j][i]

                    b = self.Hawkes_b[self.path_entity[i],self.path_entity[j]]
                    eta = self.Hawkes_eta[self.path_entity[i],self.path_entity[j]]
                    zeta = self.Hawkes_zeta[self.path_entity[i],self.path_entity[j]]

                    b_ = self.b * logit(b)
                    eta_ = self.eta * logit(eta)
                    zeta_ = self.zeta * logit(zeta)

                    s = self.cal_hawkes_likelihood_(b_, eta_, zeta_, x_ij, x_ji)
                    # print(s,'s')
                    loglike = loglike + s

                    if para_select == 0:

                        b_ = self.b * logit(new_b)
                    
                    else:

                        if para_select == 1:
                            eta_ = self.eta * logit(new_eta)
                        if para_select == 2:
                            zeta_ = self.zeta * logit(new_zeta)

                    s = self.cal_hawkes_likelihood_(b_, eta_, zeta_, x_ij, x_ji)

                    loglike_ = loglike_ + s
            
        ll = 0

        if para_select == 0:
            
            q_ = -(new_b-mu)**2/2
            q = -(self.Hawkes_b[c_send,c_receive]-mu)**2/2

            new_sample,ll = MH(loglike_, loglike, q_, q, new_b, self.Hawkes_b[c_send,c_receive])
            self.Hawkes_b[c_send,c_receive] = new_sample

        if para_select == 1:
            
            q_ = -(new_eta-mu)**2/2
            q = -(self.Hawkes_eta[c_send,c_receive]-mu)**2/2

            new_sample,ll = MH(loglike_, loglike, q_, q, new_eta, self.Hawkes_eta[c_send,c_receive])
            self.Hawkes_eta[c_send,c_receive] = new_sample

        if para_select == 2:
            
            q_ = -(new_zeta-mu)**2/2
            q = -(self.Hawkes_zeta[c_send,c_receive]-mu)**2/2

            new_sample,ll = MH(loglike_, loglike, q_, q, new_zeta, self.Hawkes_zeta[c_send,c_receive])
            self.Hawkes_zeta[c_send,c_receive] = new_sample

        # print(ll,'ll in sample_hawkes')
        return ll

def logit(x): # checked

    return 1/(1+np.exp(-x))


def MH(f_, f, q_, q, x_, x): # checked

    u = npr.uniform(low = 0.0 , high = 1.0)
    
    u_ = f_ + q_ - f - q

    if u_>0:
        return x_,f_
    elif np.exp(u_)>u:
        return x_,f_
    else:
        return x,f


def normalize_weight(weight): # checked

    # function: nomalize the log weight to 1 

    weight[:] = weight[:] - np.amax(weight)
    weight[:] = np.exp(weight[:])
    weight[:] = weight[:]/np.sum(weight)

                
    