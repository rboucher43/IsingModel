#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:05:13 2023

@author: reeseboucher
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import random


class isingModel:
        
    def __init__( self, N, exchange, T, mcSteps = 0, Tmax=4000, plotBool = False, relaxationStep = 0 ):
        '''
        Parameters
        ----------
        N : Integer
            Number of particles in lattice.
        exchange : Integer
            Determines whether lattice is ferromagnetic (exchange>0) or antiferromagnetic (exchange<0) 
        mcSteps : Integer, optional
            Number of Monte Carlo steps. The default is 0.
        plotBool : Boolean, optional
            Determines whether final lattice state is plotted or not. The default is False.
        '''
        self.N             = N
        self.mcSteps       = mcSteps
        self.exchange      = exchange
        self.T             = T
        self.Tmax          = Tmax
        self.plotBool      = plotBool
        self.relaxationStep = relaxationStep


    def updatePlane( self, lattice, position ):
        '''
        Calculates change in energy accounting for neighboring interactions with spin flip in given plane. 
        
        Parameters
        ----------
        lattice : Array
            Input arry with shape of lattice containing direction of spins
        position : Array
            Input array of position in lattice where spin will be flipped and energy will be updated
        Returns
        -------
        dE : Float
            Interaction after change in spin flip of neighboring spins

        '''
        dE = 0        
                
        lattice[position[0]][position[1]] = -1*lattice[position[0]][position[1]]

        if position[0] > 0:
            dE += -self.exchange * lattice[position[0]][position[1]] * lattice[position[0]-1][position[1]]         # up
            
            if position[1] > 0:
                dE += -self.exchange * lattice[position[0]][position[1]] * lattice[position[0]-1][position[1]-1]   # up left

            if position[1] < len(lattice[position[0]])-1:
                dE += -self.exchange * lattice[position[0]][position[1]] * lattice[position[0]-1][position[1]+1]   # up right
                
        if position[1] > 0:
            dE += -self.exchange * lattice[position[0]][position[1]] * lattice[position[0]][position[1]-1]         # left
            
        if position[1] < len(lattice[position[0]])-1:
            dE += -self.exchange * lattice[position[0]][position[1]] * lattice[position[0]][position[1]+1]         # right
            
        if position[0] < len(lattice)-1:
            dE += -self.exchange * lattice[position[0]][position[1]] * lattice[position[0]+1][position[1]]         # down
            
            if position[1] > 0:                
                dE += -self.exchange * lattice[position[0]][position[1]] * lattice[position[0]+1][position[1]-1]   # down left

            if position[1] < len(lattice[position[0]])-1:
                dE += -self.exchange * lattice[position[0]][position[1]] * lattice[position[0]+1][position[1]+1]   # down right

        return dE


     
    def planeEnergy(self, lattice):
        '''
        Calculates total energy accounting for neighboring interactions with spin flip in given plane. 

        Parameters
        ----------
        lattice : Array
            Input array with shape of lattice containing direction of spins
        Returns
        -------
        totalEnergy : Float
            Total calculated energy of input lattice accounting for neighboring interactions.
        '''
        totalEnergy = []
        for i in range(len(lattice)):
            for j in range(len(lattice[i])-1): 
                totalEnergy = np.append(totalEnergy, np.dot(-self.exchange*lattice[i][j],lattice[i][j+1])) #right
                            
        for i in range(len(lattice)-1):
            j = 1
            while j < len(lattice[i]):                 
                totalEnergy = np.append(totalEnergy,np.dot(-self.exchange*lattice[i][j], lattice[i+1][j-1]))        #low Left
                if j+1 < len(lattice[i]):
                    totalEnergy = np.append(totalEnergy,np.dot(-self.exchange*lattice[i][j], lattice[i+1][j+1]))    #low Right
                j += 2   
            for q in range(len(lattice[i])): # down 
                totalEnergy = np.append(totalEnergy,np.dot(-self.exchange*lattice[i+1][q], lattice[i][q]))      
        
        for i in range(1,len(lattice)):
            j = 1
            while j < len(lattice[i]): 
                totalEnergy = np.append(totalEnergy,np.dot(-self.exchange*lattice[i][j], lattice[i-1][j-1]))       #up Left
                if j+1 < len(lattice[i]):
                    totalEnergy = np.append(totalEnergy,np.dot(-self.exchange*lattice[i][j],lattice[i-1][j+1]))    #up Right
                j += 2  
        
        totalEnergy = np.sum(totalEnergy)
        
        return totalEnergy
    
    

    def updateLine(self, lattice, position):
        dE                = 0          
        lattice[position] = lattice[position]*-1
        
        if position < len(lattice)-1:
            dE += -self.exchange*lattice[position] * lattice[position+1]
        if position > 0:
            dE += -self.exchange*lattice[position] * lattice[position-1]
        
        return dE


    def lineEnergy(self, lattice):
        totalEnergy = []
        for site in range(len(lattice)):
            right       = lattice[site+1:1+site+1]
            totalEnergy = np.append(totalEnergy, -self.exchange*lattice[site]*right)
        totalEnergy = np.sum(totalEnergy)
        return totalEnergy
       
   
    
    def buildLattice(self, dimension, shape = None):   
        '''
        Creates lattice with shape determined by shape initialized with random spin 
        
        Parameters
        ----------
        dimension : Array
            Dimension of desired lattice
        shape : TYPE, optional
            Shape of desired lattice. The default is None. 
            If shape equals None, the lattice defaults to the shape  (dimension by ((number of atoms)\dimension))

        Returns
        -------
        lattice : Array
            Lattice with randomized spins
        '''
        if shape == None:
            shape = [dimension, int(self.N/dimension)]   
        
        lattice = np.random.randint(0, 2, self.N)
        lattice = np.where(lattice != 0, lattice, lattice-1)
        lattice = np.reshape(lattice,shape)

        return lattice
    
   
    
    def oneDimension( self ): 
        print("Beginning 1D Simulation")

        '''
        Builds 1D lattice, calculates total energy, then relaxes lattice to ground 
        state based on neighboring interaction.

        Returns
        -------
        lattice : Array
            Input arry with shape of lattice containing direction of spins.
        lowEnergy : Float
             Energy of final lattice configuration 
        '''
        lattice    = self.buildLattice( 1 )
        lattice    = lattice.flatten()
        lowEnergy  = self.lineEnergy( lattice )
        

        Tprob     = self.T/self.Tmax #probability at which spin flip will be decided
         
        energies        = []
        energiesSquared = []
        energies.append(lowEnergy)
        energiesSquared.append(lowEnergy**2)


        for step in range( int(self.mcSteps) ):
            randomPosition = np.random.randint(0,self.N)
            lattice[randomPosition] = lattice[randomPosition] * -1  

            deltaE         = self.updateLine(lattice.copy(),randomPosition)


            if deltaE < 0 or random.rand() < Tprob:
                print(lattice)
                lattice[randomPosition] = lattice[randomPosition] * -1  
                
                if step > self.relaxationStep:
                    energies.append(energies[-1]+deltaE)
                    energiesSquared.append(energies[-1]**2)


        return lattice, energies[-1]
    
    
    def twoDimensions(self, shape):
        '''
        Parameters
        ----------
        shape : Array
            Array filled with dimension of lattice

        Returns
        -------
        lattice : Array
            Input arry with shape of lattice containing direction of spins.
        averageEnergy : Float
             average energy over mc steps
        specificHeat : FLoat :
            average specific heat over mc steps

        '''
        print("Beginning 2D Simulation")

        lattice   = self.buildLattice(2, shape)
        lowEnergy = self.planeEnergy(lattice) # returns energy of initial lattice configuration
        Tprob     = self.T/self.Tmax # probability at which spin flip will be decided
         
        energies        = []
        energiesSquared = []
        energies.append(lowEnergy)
        energiesSquared.append(lowEnergy**2)

        earlyExit = False

        stepMax = 600000
        for step in range( int(self.mcSteps) ):

            if  step > stepMax and self.T < self.Tmax:
                lattice = lattice.flatten()
                np.random.shuffle(lattice)
                stepMax =  1e20               # possible future bug
                lattice = np.reshape(lattice,shape)
                lowEnergy = self.planeEnergy(lattice)
                energies = [lowEnergy]
                energiesSquared = [lowEnergy**2]

        
            randomPosition = [np.random.randint(0,shape[0]), np.random.randint(0,shape[1])]
            lattice[randomPosition[0]][randomPosition[1]] = lattice[randomPosition[0]][randomPosition[1]] * -1  # keep spin flip

            deltaE         = self.updatePlane(lattice.copy(),randomPosition)


            if deltaE < 0 or random.rand() < Tprob:

                oldlattice = lattice.copy()
                lattice[randomPosition[0]][randomPosition[1]] = lattice[randomPosition[0]][randomPosition[1]] * -1  # keep spin flip

                if np.array_equal(lattice,oldlattice) == False:
                    energies.append(energies[-1] + deltaE)
                    energiesSquared.append(energies[-1]**2)
                

                flat = lattice.flatten()

                if np.all(flat==1) or np.all(flat==-1):
                    earlyExit = True
                    self.relaxationStep = step/2
                    print(lattice)
                    break
            
                    

        if self.plotBool == True:
            self.plotSpins( lattice, shape)

        energies             = np.array(energies)
        energiesSquared      = np.array(energiesSquared)
    
              
        # these if statements are designed to only include energies after the lattice is relaxed 
        if earlyExit == False: 
            averageEnergy        = energies[int(self.relaxationStep):int(self.mcSteps)].mean()
            averageEnergySquared = energiesSquared[int(self.relaxationStep):int(self.mcSteps)].mean()
        else:
            averageEnergy        = energies[int(len(energies)/2):int(len(energies))].mean()
            averageEnergySquared = energiesSquared[int(len(energiesSquared)/2):int(len(energiesSquared))].mean()

            
        if self.T == 0: 
            specificHeat = 0
        else:
            specificHeat = (averageEnergySquared - averageEnergy**2)/((self.T))

        
        return lattice, averageEnergy, specificHeat
    
    
    
    def plotSpins(self, lattice, shape):
        '''
        Creates image of final lattice based on dimension and spin 
        Parameters
        ----------
        lattice : Array
            Input arry with shape of lattice containing direction of spins.
        shape : Array
            Shape of desired lattice.
        dimension : Integer
            Dimension of desired lattice
        '''
        lattice = np.array(lattice, dtype=float)

        # arrays filled with xy positions of spins in lattice
        xArr = np.array(np.arange(shape[0]), dtype = float)
        yArr = np.array(np.arange(shape[1]), dtype = float)

        modelName = "ISING_2d"
            
        spinup   = 0
        spindown = 0
        count    = 0
        for x in xArr:
            for y in yArr:
                count += 1
                spin = lattice[int(x)][int(y)]
                if spin > 0:
                    spinup+=1
                    plt.arrow(x,y,0,spin/1.5,width=0.001,head_width=0.25,length_includes_head=True,color=[1,0,0])
                elif spin<0:
                    plt.arrow(x,y+0.65,0,spin/1.5,width=0.001,head_width=0.25,length_includes_head=True,color=[0,0,1])
                    spindown+=1
        plt.savefig(modelName + ".png")

    
    

print("Ising Model")
# x = isingModel(400, 1, 1000,mcSteps=1e5, plotBool=True)
# x.oneDimension()
N = 400
x = isingModel(N, exchange=1, T=3600, mcSteps=1e7, plotBool=True)
output = x.twoDimensions([20,20])

magnetization = abs(output[0].sum()/N)












