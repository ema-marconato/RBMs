#Simulazione dell'ising
from __future__ import division
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import imageio
import progressbar as bar
from scipy.stats import moment


class Ising():
    def __init__(self,N):
      self.__N = N
    
    def initialstate(self):   
      ''' generates a random spin configuration for initial condition'''
      state = 2*np.random.randint(2, size=(self.__N,self.__N))-1
      return state

    def mcmove(self, config, beta):
        ''' Metropolis hasting move for N\\
            config: enter the starting Ising
        '''
        for i in range(self.__N):
            for j in range(self.__N):            
                    a = np.random.randint(0, self.__N)
                    b = np.random.randint(0, self.__N)
                    s =  config[a, b]
                    nb = config[(a+1)%self.__N,b] + config[a,(b+1)%self.__N] + config[(a-1)%self.__N,b] + config[a,(b-1)%self.__N]
                    cost = 2*s*nb
                    if cost < 0:	s *= -1
                    elif np.random.uniform() < np.exp(-cost*beta): s *= -1
                    config[a, b] = s
        return config
    
    def simulate(self, temperature, Spin_Lattice =None, boolean = False):   
        ''' This module simulates the Ising model 400 times\\
            Spin_Lattice: enter the list to add  simulated states , by default None\\
            beta: inverse of temperature\\
            boolean: set True if you want to see the plot.

        '''
        #starting lattice
        beta= 1/temperature
        config = self.initialstate()
        if boolean:
          f = plt.figure(figsize=(12, 6), dpi=80);    
          self.configPlot(f, config, 0, 1);
        
        msrmnt = 400
        for i in range(msrmnt+1):
          #calculate MH move
          self.mcmove( config, beta)

          #Random Noise on the lattice
          if i < 25 and i %2 == 0:
            for _ in range(2):
              _f , _g = np.random.randint(0,self.__N) , np.random.randint(0,self.__N)
              config[_f,_g] = -config[_f,_g]
          if i % 5== 0:
              _f , _g = np.random.randint(0,self.__N) , np.random.randint(0,self.__N)
              config[_f,_g] = -config[_f,_g]
          
          if Spin_Lattice is not None: Spin_Lattice.append(config)  
          if boolean:
            if i == 5:     self.configPlot(f, config, i,  2);
            if i == 10:    self.configPlot(f, config, i,  3);
            if i == 30:    self.configPlot(f, config, i,  4);
            if i == 50:    self.configPlot(f, config, i,  5);
            if i == 100:    self.configPlot(f, config, i,  6);
            if i == 200:   self.configPlot(f, config, i,  7);
            if i == 350:   self.configPlot(f, config, i,  8);
            
    def configPlot(self, f, config, i, n_):
        ''' This modules plts the configuration once passed to it along with time etc\\
            f: figure\\
            config: input lattice\\
            n_: subplot location
         '''
        X, Y = np.meshgrid(range(self.__N), range(self.__N))
        sp =  f.add_subplot(2, 4, n_ )  
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)      
        plt.pcolormesh(X, Y, config, cmap=plt.cm.RdBu)
        plt.title('Time=%d'%i); plt.axis('tight')    
    plt.show()
  
    def calcEnergy(self, config):
        '''Energy of a given configuration'''
        energy = 0
        for i in range(len(config)):
            for j in range(len(config)):
                S = config[i,j]
                nb = config[(i+1)%self.__N, j] + config[i,(j+1)%self.__N] + config[(i-1)%self.__N, j] + config[i,(j-1)%self.__N]
                energy += -nb*S
        return energy/4.

    def calcMag(self, config):
        '''Magnetization of a given configuration'''
        mag = np.sum(config)
        return np.abs(mag)
    
    def obs_simulation(self, eqSteps= 1000, mcSteps= 1000, tstart = 1.8, tstop=3.0, tsteps=15, fit_bool=False):
      """
      Performs a simulation of Ising 2D with eqSteps and mcSteps, with default white noise\\
      Set:\\
      tstart: temperature start\\
      tstop: temperature stop\\
      tsteps: temperature steps\\
      mag_fit: set by default True to fit the curve\\
      ene_fit: set by default True to fit the curve
      """
      temp=[]; E = []; M = []; C = []; X =[]
      n1, n2  = 1.0/(mcSteps*self.__N**2), 1.0/(mcSteps*mcSteps*self.__N**2) 
      
      for tt in np.linspace(tstart,tstop,num=tsteps):
        #print("Doing the: ",tt)
        E1 = 0; M1 = 0; E2 = 0; M2 = 0
        config = self.initialstate()
        iT=1.0/tt; iT2=iT*iT;
        
        for i in range(eqSteps):            # equilibrate
          self.mcmove(config, iT)           # Monte Carlo move
          #Random Noise after MH
          if i % 100 == 0:
            a_, b_ = np.random.randint(0,self.__N, 2)
            config[a_,b_] = -config[a_,b_]

        magnet_vec = []
        energy_vec = []
        for i in range(mcSteps):
          self.mcmove(config, iT)        
          #Random Noise after MH   
          if i % 100 == 0:
            a_, b_ = np.random.randint(0,self.__N, 2)
            config[a_,b_] = -config[a_,b_]

          Ene = self.calcEnergy(config)     # calculate the energy
          Mag = self.calcMag(config)        # calculate the magnetisation
          magnet_vec.append(Mag/self.__N**2)
          energy_vec.append(Ene/self.__N**2)

          E1 += Ene
          M1 += Mag
          M2 += Mag**2
          E2 += Ene**2
          
        
        mean_M  = np.mean(magnet_vec)
        mean_E  = np.mean(energy_vec)
        sigma_M = moment(magnet_vec, moment=2)
        sigma_E = moment(energy_vec, moment=2)
        
        mean_C  = self.__N**2/tt**2 * sigma_E
        mean_X  = self.__N**2/tt**2 * sigma_M
        sigma_C = self.__N**4/tt**4 * moment(energy_vec, moment = 4) -mean_C**2
        sigma_X = self.__N**4/tt**4 * moment(magnet_vec, moment = 4)- mean_X**2

        print("| At Temperature =%1.2f"%tt," <E>= %1.3f"%mean_E," <M>= %1.3f"%mean_M)

        temp.append(tt); 
        E.append([mean_E, np.sqrt(sigma_E**2)]);  M.append([mean_M, np.sqrt(sigma_M**2)]);  
        C.append([mean_C,np.sqrt(sigma_C)]);   X.append([mean_X, np.sqrt(sigma_X)])

      E = np.asarray(E)
      M = np.asarray(M)
      C = np.asarray(C)
      X = np.asarray(X)
      f = plt.figure(figsize=(15, 8)); # plot the calculated values    
      
      sp =  f.add_subplot(2, 2, 1 );
      if fit_bool: 
        plt.errorbar(temp, E[:,0],  yerr=E[:,1], linestyle="", color="orange", marker="x", ecolor="orange", capsize=8, label="Energy from MH")
        x = np.asarray(temp); y = E[:,0]; sigma_y =(E[:,1]); E_max = np.max(E[:,0]) +0.01; E_min = np.min(E[:,0]) -0.01
        A,B = np.polyfit(x, np.log((E_min-E_max)/(y-E_max) -1), deg=1)
        x = np.linspace(1.8,3,1000)
        print("Expected from Energy Tc: ", np.abs(B/A))
        plt.plot(x, (E_min-E_max)/(1+np.exp(A*x+B)) + E_max,color="gray", linestyle="--", label="Fitted curve")
      else:              
        plt.plot(temp, E[:,0], linestyle="", color="orange", marker="x", label="Energy from MH")
      plt.xlabel("Temperature (T)");
      plt.ylabel("Energy ");         plt.axis('tight');
      plt.legend()


      sp =  f.add_subplot(2, 2, 2 );
      if fit_bool: 
        plt.errorbar(temp, np.abs(M[:,0]), yerr=M[:,1], linestyle="", color="green", marker="x", ecolor="green",capsize=8, label="Magnetization from MH")
        x = np.asarray(temp); y = np.abs(M[:,0]); sigma_y =(M[:,1]); 
        A,B = np.polyfit(x, np.log(1/y -1), deg=1)
        print("Expected from Magnetization Tc: ", np.abs(B/A))
        x = np.linspace(1.8,3,1000)
        plt.plot(x, 1/(1+np.exp(A*x+ B)),color="gray", linestyle="--", label="Fitted curve")
      else:
        plt.plot(temp, np.abs(M[:,0]), linestyle="", color="green", marker="x", label="Magnetization from MH")
      plt.xlabel("Temperature (T)"); 
      plt.ylabel("Magnetization ");   plt.axis('tight');
      plt.legend()

      sp =  f.add_subplot(2, 2, 3 );
      if fit_bool:
        plt.errorbar(temp, C[:,0],  yerr=C[:,1], linestyle="", color="red", marker="x", ecolor="red", capsize=8, label="Specifiv Heat from MH")
      else:
        plt.plot(temp, C[:,0], linestyle="", color="red", marker="x", label="Specific Heat from MH")
      plt.xlabel("Temperature (T)");  
      plt.ylabel("Specific Heat ");   plt.axis('tight');   
      plt.legend()

      sp =  f.add_subplot(2, 2, 4 );
      if fit_bool:        
        plt.errorbar(temp, X[:,0],  yerr=X[:,1], linestyle="", color="blue", marker="x", ecolor="blue", capsize=8, label="Susceptibility Heat from MH")
      else:
        plt.plot(temp, X[:,0],  linestyle="", color="blue", marker="x", label="Susceptibility Heat from MH")
      plt.xlabel("Temperature (T)"); 
      plt.ylabel("Susceptibility");   plt.axis('tight');
      plt.legend()

      return temp, E, M, C, X
