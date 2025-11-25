__authors__ = "Zachary Klouchnikov and Hannah Semple"

# HEADER

"""
IMPORTS
"""
import random
import numpy as np
import vpython as vp
import matplotlib.pyplot as plt

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

"""
FUNCTIONS
"""
def mag(x: np.ndarray) -> float:
    """Returns the magnitude of a 2D vector x.
    Argumets:
    x -- 2D vector
    """

    return np.sqrt(x[0] ** 2 + x[1] ** 2)

# Function to calculate the total length of the tour
def distance():
    """Returns the total distance of the tour based on the current city
    locations.
    """

    s = 0.0
    for i in range(N):
        s += mag(r[i + 1] - r[i])

    return s

"""
PART A)
"""
"Constants"
N = 25
R = 0.02
TMAX = 10.0
TMIN = 1e-3
NS = [1, 2, 3, 4, 5]

"Initialize city locations"
random.seed(NS[0])

r = np.empty([N + 1, 2], dtype = float)
for i in range(N):
    r[i, 0] = random.random()
    r[i, 1] = random.random()
r[N] = r[0]
d = distance()

"Simulated annealing for tau = 1e4"
tau = 1e4
tau_default = []

for i in range(len(NS)):
    random.seed(NS[i])

    t = 0
    t_index = TMAX
    while t_index > TMIN:

        # Cooling
        t += 1
        t_index = TMAX * np.exp(-t / tau)

        # Choose two cities to swap and make sure they are distinct
        i, j = random.randint(1, N), random.randint(1, N)
        while i == j:
            i, j = random.randint(1, N), random.randint(1, N)

        # Swap them and calculate the change in distance
        oldD = d
        r[i, 0], r[j, 0] = r[j, 0], r[i, 0]
        r[i, 1], r[j, 1] = r[j, 1], r[i, 1]
        d = distance()
        deltaD = d - oldD

        # If the move is rejected, swap them back again
        if random.random() > np.exp(-deltaD / t_index):
            r[i, 0], r[j, 0] = r[j, 0], r[i, 0]
            r[i, 1], r[j, 1] = r[j, 1], r[i, 1]
            d = oldD

    tau_default.append(d)

"Simulated annealing for tau = 1e3"
tau = 1e3
tau_lower = []

for i in range(len(NS)):
    random.seed(NS[i])

    t = 0
    t_index = TMAX
    while t_index > TMIN:

        # Cooling
        t += 1
        t_index = TMAX * np.exp(-t / tau)

        # Choose two cities to swap and make sure they are distinct
        i, j = random.randint(1, N), random.randint(1, N)
        while i == j:
            i, j = random.randint(1, N), random.randint(1, N)

        # Swap them and calculate the change in distance
        oldD = d
        r[i, 0], r[j, 0] = r[j, 0], r[i, 0]
        r[i, 1], r[j, 1] = r[j, 1], r[i, 1]
        d = distance()
        deltaD = d - oldD

        # If the move is rejected, swap them back again
        if random.random() > np.exp(-deltaD / t_index):
            r[i, 0], r[j, 0] = r[j, 0], r[i, 0]
            r[i, 1], r[j, 1] = r[j, 1], r[i, 1]
            d = oldD

    tau_lower.append(d)

"Simulated annealing for tau = 1e5"
tau = 1e5
tau_higher = []

for i in range(len(NS)):
    random.seed(NS[i])

    t = 0
    t_index = TMAX
    while t_index > TMIN:

        # Cooling
        t += 1
        t_index = TMAX * np.exp(-t / tau)

        # Choose two cities to swap and make sure they are distinct
        i, j = random.randint(1, N), random.randint(1, N)
        while i == j:
            i, j = random.randint(1, N), random.randint(1, N)

        # Swap them and calculate the change in distance
        oldD = d
        r[i, 0], r[j, 0] = r[j, 0], r[i, 0]
        r[i, 1], r[j, 1] = r[j, 1], r[i, 1]
        d = distance()
        deltaD = d - oldD

        # If the move is rejected, swap them back again
        if random.random() > np.exp(-deltaD / t_index):
            r[i, 0], r[j, 0] = r[j, 0], r[i, 0]
            r[i, 1], r[j, 1] = r[j, 1], r[i, 1]
            d = oldD

    tau_higher.append(d)

"Plotting Simulated Annealing Optimization"
plt.figure()

# Plotting simulated annealing optimization
plt.plot(NS, tau_default, marker = 'o', color = 'Teal', label = "Final Tour Distances for $\\tau = 1e4$")
plt.plot(NS, tau_lower, marker = 'o', color = 'Coral', label = "Final Tour Distances for $\\tau = 1e3$")
plt.plot(NS, tau_higher, marker = 'o', color = 'Purple', label = "Final Tour Distances for $\\tau = 1e5$")

# Labels
plt.title("Simulated Annealing Optimization", fontsize = 12)
plt.xlabel("Seed Value", fontsize = 12)
plt.ylabel("Final Tour Distance", fontsize = 12)

plt.xticks(NS)

plt.legend(fontsize = 12)
plt.grid()

# Limits
plt.xlim(1, 5)

plt.savefig('Figures\\Simulated Annealing Optimization.pdf')
plt.show()
