import torch
import numpy as np
import progressbar as bar

class RBM():
  def __init__(self, nv, nh):
    self.__nv = nv
    self.__nh = nh
    self.W = torch.randn(self.__nh, self.__nv).float()
    self.c = torch.randn(self.__nh,).float()
    self.b = torch.randn(self.__nv,).float() 
      
  def sample_h(self, x, beta =1):
    wx = torch.mm(x, self.W.t())
    activation = wx + self.c   #self.c.expand_as(wx)
    #print(beta, activation)
    p_h_given_v = torch.sigmoid(beta*activation)
    return p_h_given_v, torch.bernoulli(p_h_given_v)

  def sample_v(self, y, beta=1):
    wy = torch.mm(y, self.W)
    activation = wy + self.b   #self.b.expand_as(wy)
    #print(beta, activation)
    p_v_given_h = torch.sigmoid(beta*activation)
    return p_v_given_h, torch.bernoulli(p_v_given_h)

  def train(self, lr, delta_W, delta_b, delta_c):
    self.W += lr* delta_W
    self.b += lr* delta_b
    self.c += lr* delta_c



#######################################################################################
###             EXTENDED RBM CLASS TO PROVIDE AIS/GIBBS/METROPOLIS SAMPLES          ###
#######################################################################################

class Rbm(RBM):
    def __init__(self, nv, nh, T, W = None, b = None, c = None):
        RBM.__init__(self,nv,nh)
        self._T = T
        self._L =int(np.sqrt(nv))
        if W is not None: self.W = W
        if b is not None: self.b = b
        if c is not None: self.c = c

    def AIS(self, tsteps=1000, n_samples=1000):
      samples = torch.randint(2,size = (n_samples, self._L**2)).float()
      for tt in np.linspace(0.001, 1, 1000):
        _, h       = self.sample_h(samples, tt)
        _, samples = self.sample_v(h, tt)
      return samples

    def AIS_sample(self, n_samples=1000, print_means=False):
        """ 
        Based on AIS, it returns a matrix of size=(n_samples, 4) with the observables for the Ising Model
        """
        samples = self.AIS(self, n_samples=n_samples)
        samples[samples == 0] = -1
        mean_e, mean_m, mean_h, mean_s = self.calculate_observables(samples, n_samples)
        if print_means: print("Energy: "+str(mean_e)+" Magnetization: "+str(mean_m)+" Specific Heat: "+str(mean_h)+" Susceptibility: "+str(mean_s))
        return samples, [mean_e, mean_m, mean_h, mean_s]


    def Gibbs_step(self, state, k_moves, return_states=False):
      states = torch.zeros(size=(k_moves, self._L**2))
      for k in range(k_moves):
        _, h      = self.sample_h(state.float())
        _, state  = self.sample_v(h)
        states[k] = state
      if return_states: return states
      else:             return state
    
    def Gibbs_sample(self, k_CD = 1, eqsteps=1000, n_samples=1000, print_means=False):
        """ 
        Based on Gibbs-S, it returns a matrix of size=(n_samples, 4) with the observables for the Ising Model
        """
        sample  = torch.randint(2,size=(1,self._L**2)) 
        sample  = self.Gibbs_step(sample, eqsteps)
        samples = self.Gibbs_step(sample, n_samples, True)
        samples[samples == 0] = -1
        
        mean_e, mean_m, mean_h, mean_s = self.calculate_observables(samples, n_samples)
        if print_means: print("Energy: "+str(mean_e)+" Magnetization: "+str(mean_m)+" Specific Heat: "+str(mean_h)+" Susceptibility: "+str(mean_s))
        return samples,  [mean_e, mean_m, mean_h, mean_s]
    

    def Metropolis_step(self, state):
      wstate = np.asarray(state)
      
      for _ in range(self._L**2):
        i = np.random.randint(0,high=self._L**2)
        new_state    = wstate
        #print("Location:",i )
        #print("Before \n",new_state)

        if new_state[i] == 0:
          new_state[i] = 1
          #print("After \n",new_state)
        else: 
          new_state[i] = 0
          #print("After \n",new_state)
        #new_state[i] = torch.tensor(1).float() -wstate[i].float()
        #print(torch.norm( (wstate-new_state).float() ))
        
        energy0      = self.free_energy(state.float())
        energy1      = self.free_energy(torch.tensor(new_state).float())
        u            = np.random.uniform()
        
        #prob = self.free_energy((new_state-state).float())
        #print("Free energy difference: ", torch.exp(prob))
        prob2 = torch.exp(energy1-energy0)
        #print("Prob2 as normal: ", prob2," and u:", u)
        if min(1, prob2.item()) > u:
          if np.random.uniform() > 0.5:
            wstate = new_state
          print("Changed in ",i)
          #print( torch.norm((wstate-state).float() ))
      print("Final state:", wstate)
      return wstate  

    def Metropolis_sample(self, eqsteps= 1000, n_samples=1000, print_means=False):
      state = np.random.randint(0, high=2, size=self._L**2)

      state = torch.tensor(state)
      for _ in range(eqsteps): state = self.Metropolis_step(state)

      configurations = torch.zeros(size=(n_samples,self._L**2))
      check = True
      for k in bar.progressbar(range(n_samples)):
        state = self.Metropolis_step(state)
        configurations[k] = state
        
      configurations[configurations==0] = -1
      mean_e, mean_m, mean_h, mean_s = self.calculate_observables(configurations, n_samples)
      if print_means: print("Energy: "+str(mean_e)+" Magnetization: "+str(mean_m)+" Specific Heat: "+str(mean_h)+" Susceptibility: "+str(mean_s))
      return configurations,  [mean_e, mean_m, mean_h, mean_s]
    

    #IMPORTING THE FUNCTION FROM MAIN MODULE
    def calculate_observables(self, Spin_Lattice, how_many=10000):

      '''This is the implementation to calculate the observables on the Ising Lattice \n
        Spin_Lattice:  insert the Tensor on which to calculate obs \n
        temperature:   set the temperature of the model   \n
        how_many:      the number of datapoints to use  \n
      '''
      torch.set_default_dtype(torch.float32)
      Spin_Lattice = Spin_Lattice[torch.randperm(len(Spin_Lattice))]
      spin_configuration = Spin_Lattice[0:how_many].float()
      spin_configuration[spin_configuration == 0] = -1

      #mag = np.abs(torch.einsum('ch,ch->',  torch.ones(how_many,36), spin_configuration)/(36*how_many))
      ene, mag, ene2, mag2 = torch.tensor(0).float(), torch.tensor(0).float(), torch.tensor(0).float(), torch.tensor(0).float()
      for lattice in spin_configuration:    
        #print("Type mag: ", mag.type(), " type addendum: ", torch.abs(torch.dot(torch.ones(L**2), lattice.float())).type() )
        #print("a: ", torch.ones(L**2).size(), " b: ", lattice.size() )
        mag   += torch.abs(torch.dot(torch.ones(self._L**2), lattice.float()))
        mag2  += torch.abs(torch.dot(torch.ones(self._L**2), lattice.float()))**2
        energy = torch.tensor(0).float()
        lattice= lattice.reshape(self._L,self._L)
        for i in range(self._L):
          for j in range(self._L):
            energy += -lattice[i,j]*(lattice[(i+1)%self._L, j] + lattice[i,(j+1)%self._L] + lattice[(i-1)%self._L, j] + lattice[i,(j-1)%self._L])
          
        ene  += energy/how_many     
        ene2 += energy**2
      Energy = ene/(self._L**2)/4
      Magnet = mag/(self._L**2)/how_many
      Spec_H = 1/(self._T**2)*(ene2/how_many/(self._L**2) - ene**2/(self._L**2))/16
      Susc   = 1/(self._T**2)*(mag2/how_many/(self._L**2) - mag**2/(self._L**2)/(how_many**2)) 

      return Energy, Magnet, Spec_H, Susc

    #IMPORTING THE FUNCTION FROM MAIN MODULE
    def free_energy(self, v, beta=1):
      term_1      = torch.exp(beta*torch.dot(self.b, v))                                #scalar
      product_w_c = 1 + torch.exp(beta*self.c + beta*torch.mv(self.W, v))   
      product_w_c = torch.log(product_w_c) 
      term_2      = torch.sum(product_w_c.float())  
      #term_2 = torch.max(product_w_c)                                                                                 #scalar 
      #return torch.mul(term_1,term_2)
      return torch.log(term_1) + term_2  




############################################################################################################
###                                    Container of RBMs                                                 ###
############################################################################################################

class mother_rbm:
    def __init__(self,nv,nh,**kwargs):
        self.__L = int(np.sqrt(nv)) 
        self.__nv = nv
        self.__nh = nh
        self.machine = {} #defining a dictionary of rbms
        for label,obj in kwargs.items():
            self.machine[str(label)] = Rbm(nv,nh, T = obj[0], W = obj[1], b = obj[2], c = obj[3])
        print("# Istantiated ", len(self.machine), " Rbms into dictionary.")
    
    def add(self, **kwargs):
        for label, obj in kwargs.items():
            self.machine[str(label)] = Rbm(self.__nv,self.__nh, T = obj[0], W = obj[1], b = obj[2], c = obj[3])
            print("Added",label,"in mother_rbm.")
        
    def calculate_state(self, state):
        state[state==0]=-1
        state = np.asarray(state).reshape(self.__L,self.__L)
        mag = np.sum(state)/self.__L**2
        ene = 0
        for i in range(self.__L):
            for j in range(self.__L):
                s    = state[i,j]
                nb   = state[(i+1)%self.__L, j] +state[(i-1)%self.__L, j] +state[i, (j+1)%self.__L] +state[i, (j-1)%self.__L]
                ene += -nb*s/self.__L**2
        return np.abs(mag), ene/4
    
    def simulate_AIS(self, n_samples= 1000):
      """
      Enter number of samples: n_samples\\
      Returns: obs: array of mean obs, ene: array of energy per state, mag: array of magnetization per state
      """
      all_obs , energy, magnet = [], [], []
      
      for label in self.machine:
          samples, obs = self.machine[label].AIS_sample(n_samples )
          all_obs.append(obs)
          single_obs = []
          for state in samples: 
              mag, ene = self.calculate_state(state)
              single_obs.append([mag,ene])
          
          energy.append(np.asarray(single_obs)[:,1])
          magnet.append(np.asarray(single_obs)[:,0])
      return np.asarray(all_obs), np.asarray(energy), np.asarray(magnet)

    def simulate_GIB(self, n_samples= 1000):
      """
      Enter number of samples: n_samples\\
      Returns: obs: array of mean obs, ene: array of energy per state, mag: array of magnetization per state
      """
      all_obs , energy, magnet = [], [], []
      
      for label in self.machine:
          samples, obs = self.machine[label].Gibbs_sample(n_samples)
          all_obs.append(obs)
          single_obs = []
          for state in samples: 
              mag, ene = self.calculate_state(state)
              single_obs.append([mag,ene])
          
          energy.append(np.asarray(single_obs)[:,1])
          magnet.append(np.asarray(single_obs)[:,0])
      return np.asarray(all_obs), np.asarray(energy), np.asarray(magnet)


    def __len__(self):
      return len(self.machine)
    
    def __repr__(self):
      return f"Object mother_rbm for lattice {self.__L}**2, with labels {[label for label in self.machine]}"

