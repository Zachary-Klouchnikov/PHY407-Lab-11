__authors__ = "Zachary Klouchnikov and Hannah Semple"

# This program calculates the total energy and magnetization for a 1D Ising model with N dipoles
# Adapted from: Nico Grisouard, University of Toronto


"""
IMPORTS
"""
import numpy as np
from random import random, randrange, randint
import matplotlib.pyplot as plt

"""
FUNCTIONS
"""
def energy_function(arr, J_=1):
    """
    Calculates the energy of the system where the input is a 20x20 dipole array
    """
    row, collumn = np.zeros([1, 20], float), np.zeros(20, float)
    for i in range(0, 20 - 1):
        collumn += -J_ * arr[:, i] * arr[:, i + 1]
        row += -J_ * arr[i, :] * arr[i + 1, :]
    E = sum(row[0]) + sum(collumn)
    return E


def acceptance(Enew, Eold, T):
    """
    Returns True or False whether the MC move will be accepetd
    """
    beta = 1/T
    dE = Enew - Eold
    if dE >= 0:
        P = np.exp(- beta * dE)
    elif dE < 0:
        return True
    return False if random() > P else True

"""
PART A,B,C,D
"""
# define constants
kB = 1.0
T = 1.0
J = 1.0
num_dipoles = 400  #20*20 = 400
N = int(1e5)

dipoles = np.random.rand(20,20)  #initialise dipole array where every element is randomly in [0,1)
for j in range(20):   #assigning each elemnt +-1 based on its random value   
    for i in range(20):
        if dipoles[j][i] < 0.5:
            dipoles[j][i] = -1
        else:
            dipoles[j][i] = 1


energy = []  # empty list; to add to it, use energy.append(value)
magnet = []  # empty list; to add to it, use magnet.append(value)


for i in range(N):  #go through each step
    Eold = energy_function(dipoles)
    picked_i, picked_j = randrange(20), randrange(20)  # choose a victim
    dipoles[picked_i, picked_j] *= -1  # propose to flip the victim
    Enew = energy_function(dipoles)  # compute Energy of proposed new state

    # calculate acceptance probability
    accepted = acceptance(Enew, Eold, T)
    if accepted:
        energy.append(Enew)
    else:
        dipoles[picked_i, picked_j] *= -1 
        energy.append(Eold)
    magnet.append(np.sum(dipoles))
    

plt.figure()
plt.plot(magnet, color='teal')
plt.xlabel('Step Number')
plt.ylabel('Total Magnetisation')
plt.title('Total Magnetisation Evolution')
plt.grid()
# plt.savefig('Q4c.pdf', bbox_inches='tight')
plt.show()


"""
PART E
"""
kB = 1.0
Ts = [1.0, 2.0, 3.0]
J = 1.0
num_dipoles = 400  #20*20 = 400
N = int(1e5)

magnets = []

for T in Ts:
    dipoles = np.random.rand(20,20)  #initialise dipole array where every element is randomly in [0,1)
    for j in range(20):   #assigning each elemnt +-1 based on its random value   
        for i in range(20):
            if dipoles[j][i] < 0.5:
                dipoles[j][i] = -1
            else:
                dipoles[j][i] = 1

    energy = []  # empty list; to add to it, use energy.append(value)
    magnet = []  # empty list; to add to it, use magnet.append(value)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    
    for i in range(N):  #go through each step
        Eold = energy_function(dipoles)
        picked_i, picked_j = randrange(20), randrange(20)  # choose a victim
        dipoles[picked_i, picked_j] *= -1  # propose to flip the victim
        Enew = energy_function(dipoles)  # compute Energy of proposed new state

        # calculate acceptance probability
        accepted = acceptance(Enew, Eold, T)
        if accepted:
            energy.append(Enew)
        else:
            dipoles[picked_i, picked_j] *= -1 
            energy.append(Eold)
        magnet.append(np.sum(dipoles))
        
        if i==0:
            ax1.imshow(dipoles, cmap='bwr',alpha=0.75)
            ax1.set_title('Step Number {}'.format(i+1))
        
        if i==(int(0.5e5)):
            ax2.imshow(dipoles, cmap='bwr',alpha=0.75)
            ax2.set_title('Step Number {}'.format(i))
            
        if i==(int(1e5)-1):
            ax3.imshow(dipoles, cmap='bwr',alpha=0.75)
            ax3.set_title('Step Number {}'.format(i+1))
        
    magnets.append(magnet)
    plt.tight_layout()
    # plt.savefig('Q4e_{}.pdf'.format(T), bbox_inches='tight')
    plt.show()


plt.figure()
plt.plot(magnets[0], color='teal', label='T=1.0')
plt.plot(magnets[1], color='purple', label='T=2.0')
plt.plot(magnets[2], color='coral', label='T=3.0')
plt.xlabel('Step Number')
plt.ylabel('Total Magnetisation')
plt.title('Total Magnetisation Evolution')
plt.grid()
plt.legend()
# plt.savefig('Q4e_Ts.pdf', bbox_inches='tight')
plt.show()    

