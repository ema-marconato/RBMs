from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
import tensorflow as tf
import torch
import time
import progressbar

torch.set_default_dtype(torch.float32)

class txt_Ising_dataset(Dataset):
    """ Defines a txt reader """
    def __init__(self, csv_file, size=10**5, L=8, transform=None, skiprows=1):
        self.csv_file = csv_file
        self.size = size
        csvdata = np.loadtxt(csv_file, delimiter=",", skiprows=skiprows, dtype="float32")[0:L*size]
        self.imgs = torch.from_numpy(csvdata.reshape((size,L**2)) )
        self.datasize, sizesq = self.imgs.shape
        self.transform = transform
        print("# Loaded training set of %d states" % self.datasize)

    def __getitem__(self, index):
        return self.imgs[index], index

    def __len__(self):
        return len(self.imgs)

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

def calculate_observables(Spin_Lattice, temperature, L, how_many=10000):

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
    mag   += torch.abs(torch.dot(torch.ones(L**2), lattice.float()))
    mag2  += torch.abs(torch.dot(torch.ones(L**2), lattice.float()))**2
    energy = torch.tensor(0).float()
    lattice= lattice.reshape(L,L)
    for i in range(L):
      for j in range(L):
        energy += -lattice[i,j]*(lattice[(i+1)%L, j] + lattice[i,(j+1)%L] + lattice[(i-1)%L, j] + lattice[i,(j-1)%L])
      
    ene  += energy/how_many     
    ene2 += energy**2
  Energy = ene/(L**2)
  Magnet = mag/(L**2)/how_many
  Spec_H = 1/(temperature**2)*(ene2/how_many/(L**2) - ene**2/(L**2)/(how_many**2))
  Susc   = 1/(temperature**2)*(mag2/how_many/(L**2) - mag**2/(L**2)/(how_many**2)) 

  return Energy, Magnet, Spec_H, Susc


def AIS(rbm, how_many, print_status = False):
  dataset = torch.randint(2, size=(how_many, len(rbm.b)) ).float()
  for tt in np.linspace(0.001, 1, 1000):
    _, h_0     = rbm.sample_h(dataset, tt)
    _, dataset = rbm.sample_v(h_0, tt)
  if print_status: print("# Created database AIS with size ",how_many)
  return dataset

def GibbS(rbm, k_sampling, how_many, print_status = False):
  dataset = torch.randint(2,size=(how_many, len(rbm.b) )).float()
  _iter = 0	
  while _iter < k_sampling:
    _, v_k     = rbm.sample_v(dataset)
    _, dataset = rbm.sample_h(v_k)
    _iter     += 1
  if print_status: print("# Created database GibbS with size ",how_many)
  return v_k


def sort_from_rbm(rbm,temp, observables_AIS, observables_GIBBS, how_many, L):
  v_k = AIS(rbm, how_many)
  v_j = GibbS(rbm, 1, how_many)
  v_k[v_k ==0] = -1
  v_j[v_j ==0] = -1
  mean_e, mean_m, mean_h, mean_s = calculate_observables(v_k, temp, L, how_many)
  mean_e_G, mean_m_G, mean_h_G, mean_s_G = calculate_observables(v_j, temp, L, how_many)
  observables_AIS.append([mean_e,mean_m,mean_h,mean_s])
  observables_GIBBS.append([mean_e_G, mean_m_G, mean_h_G, mean_s_G])
  #print("Energy: "+str(mean_e)+" Magnetization: "+str(mean_m)+" Specific Heat: "+str(mean_h)+" Susceptibility: "+str(mean_s))

def cast_to_torch(vector, l):
  vector = np.asarray(vector).reshape(l,36)
  np.random.shuffle(vector)
  vector = torch.from_numpy(vector).float()
  return vector



def train_rbm(rbm, Spin_Config, temp, nb_epoch, k_CD, lr, L, print_log = 1000, reg=False, gamma=0.9):
  #ep_time = current_milli_time()
  no_configurations = len(Spin_Config)
  batch_size = 200
  #observables_AIS = []
  #observables_GIBBS = []
  log_lh_data = []
  loss = []
  for epoch in progressbar.progressbar(range(1, nb_epoch + 1)) :
      train_loss = 0
      s = 0.
      Spin_Config = Spin_Config[torch.randperm(no_configurations)]

      for id_config in range(0, no_configurations - batch_size, batch_size):
        v_0 = Spin_Config[id_config:id_config + batch_size]
        #CD k=k_CD implementation
        r = 0
        p_h_given_v, h_k = rbm.sample_h(v_0) 
        while (r < k_CD):
          p_v_given_h, v_k   = rbm.sample_v(h_k)
          p_h_given_v_k, h_k = rbm.sample_h(v_k)
          r += 1

        #SGD implementation
        delta_W = torch.zeros(size = [batch_size] + list(rbm.W.size()) ).float()
        delta_b = torch.zeros(size = [batch_size] + list(rbm.b.size()) ).float()
        delta_c = torch.zeros(size = [batch_size] + list(rbm.c.size()) ).float()
    
        for i_ in range(batch_size):
          delta_W[i_]    += torch.ger(p_h_given_v[i_], v_0[i_]) - torch.ger(p_h_given_v_k[i_], v_k[i_])
        delta_b          += (v_0 - v_k)
        delta_c          += p_h_given_v - p_h_given_v_k 
        
        if reg:  rbm.train(lr = lr, 
                  delta_W = torch.einsum('c,cnh->nh',  torch.ones(batch_size), delta_W)/batch_size - gamma*rbm.W, 
                  delta_b = torch.einsum('c,ch->h',    torch.ones(batch_size), delta_b)/batch_size - gamma*rbm.b, 
                  delta_c = torch.einsum('c,ch->h',    torch.ones(batch_size), delta_c)/batch_size - gamma*rbm.c)

        else: rbm.train(lr = lr, 
                        delta_W = torch.einsum('c,cnh->nh',  torch.ones(batch_size), delta_W)/batch_size, 
                        delta_b = torch.einsum('c,ch->h',    torch.ones(batch_size), delta_b)/batch_size, 
                        delta_c = torch.einsum('c,ch->h',    torch.ones(batch_size), delta_c)/batch_size )

        
        s += 1.
        train_loss += 1/batch_size*torch.norm(v_0 - v_k) 
      train_loss /= s
      if epoch % print_log == 0:
        #sort_from_rbm(rbm, temp, observables_AIS, observables_GIBBS, 1000, L=L)
        Log_error =   log_lh(rbm, Spin_Config, L=L)
        log_lh_data.append(Log_error)
        print(" | Log_LH: ", Log_error, " | Loss: ", train_loss)
        loss.append(train_loss)
      else: print(" | Loss: ", train_loss)

      #print('epoch: '+str(epoch)+ " loss: "+str(train_loss) +" --delta time: "+str(ep_sec_time) +" seconds")
      #ep_time = current_milli_time() 
  return log_lh_data, loss


def free_energy(rbm, v, L,beta=1):
  term_1      = torch.exp(beta*torch.dot(rbm.b, v))                                #scalar
  product_w_c = torch.ones(L**2) + torch.exp(beta*rbm.c + beta*torch.mv(rbm.W, v)) #vector
  term_2      = torch.prod(product_w_c.float())                                    #scalar 
  return torch.mul(term_1,term_2)

def free_energy_fraction(rbm,v, beta,steps=1000):
  term_1 = torch.exp(1/steps*torch.dot(rbm.b, v))                                #scalar
  term_2 = (1 + torch.exp(beta*rbm.c + beta*torch.mv(rbm.W, v)))                #scalar 
  term_3 = (1 + torch.exp((beta-1/steps)*rbm.c + (beta-1/steps)*torch.mv(rbm.W, v)))  
  term_4 = torch.prod(torch.div(term_2,term_3))
  try:
    energy = torch.mul(term_1,term_4) 
  except:
    print("Error in energy valuation!"); 
    return None
  return energy

def log_Zeta(rbm, L, steps= 1000):
  #starting with random Ising
  container = torch.randint(1,size=(1000,L**2)).float()
  beta_1 = 0.0001
  log_z1 = 0 
  
  #calculate z1
  for i in range(200): log_z1 += free_energy(rbm, container[i], L, beta_1)/200
  log_z1 = torch.log(log_z1)
  
  #main cycle
  Log_Z = log_z1 
  
  for tt in np.linspace(0.002,1.,steps-1):
    container = container[torch.randperm(1000)]
    
    #Gibbs sampling k = 1, to proceed with AIS
    _, h_0       = rbm.sample_h(container, tt)
    _, container = rbm.sample_v(h_0, tt)
    z_j   = torch.tensor(0).float()
    for j in range(50):
       z_j += free_energy_fraction(rbm, container[j], tt, steps)/50 
    Log_Z += torch.log(z_j)
  return Log_Z + L**2*np.log(2)

def log_lh(rbm, spins,L, nsamples= 10000):
  log_Z = log_Zeta(rbm, L)
  log_p_star = 0
  _, h = rbm.sample_h(spins)
  for j in range(min(nsamples,  len(spins))):
    try: 
      pre_log = log_p_star
      log_p_star += torch.dot(rbm.b, spins[j])/nsamples + torch.sum(torch.log(1+  torch.exp(rbm.c + torch.mv(rbm.W, spins[j]) ) ) )/nsamples 
#      if j % 1000==0: print("In cycle ",j," with free_energy_log: ", log_p_star )
    except: 
#      print("Error in: ",j," cycle.")
      return 0
  return log_p_star - log_Z
