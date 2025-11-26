__authors__ = "Zachary Klouchnikov and Hannah Semple"

# The following code performs simulated annealing to solve two problems:
# Part A) The Traveling Salesman Problem (TSP) for 25 cities with different cooling schedules
# Part B) Optimization of a given function using simulated annealing techniques.
# Code from Part A) use code from salesman.py provided by Newman.

"""
IMPORTS
"""
import random
import numpy as np
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

def gaussian():
    """Returns a random number drawn from a Gaussian distribution.
    """
    x = np.random.rand()
    y = np.random.rand()

    return np.sqrt(-2 * np.log(x)) * np.cos(2 * np.pi * y)

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

# plt.savefig('Figures\\Simulated Annealing Optimization.pdf')
plt.show()

"""
PART B)
"""
"Constants and defining functions"
T_0 = 1.0
T_MIN = 1e-3
TAU = 1e4

# Equation (10)
f = lambda x, y: (x ** 2) - np.cos(4 * np.pi * x) + (y - 1) ** 2
# Equation (12)
g = lambda x, y: np.cos(x) + np.cos(np.sqrt(2) * x) + np.cos(np.sqrt(3) * x) + (y - 1) ** 2

"Simulated annealing to minimize equation (10)"
t = 0
x = 2.0
y = 2.0

path_x = []
path_y = []
path_f = []

t_index = T_0
while t_index > T_MIN:
    path_x.append(x)
    path_y.append(y)
    path_f.append(f(x, y))

    # Cooling
    t += 1
    t_index = T_0 * np.exp(-t / TAU)

    # Propose Monte Carlo step
    dx = gaussian()
    dy = gaussian()
    x_new = x + dx
    y_new = y + dy

    # Energy difference
    dE = f(x_new, y_new) - f(x, y)

    # Metropolis probability
    if dE < 0 or np.random.rand() < np.exp(-dE / t_index):
        x, y = x_new, y_new

print(f"Final position: x = {x}, y = {y}")

"Plotting Simulated Annealing Optimization of Equation (10)"
plt.figure()

# Plotting simulated annealing optimization of Equation (10)
plt.scatter(path_x, path_y, c = np.arange(len(path_x)), cmap = 'viridis', s = 4)

# Labels
plt.title("Simulated Annealing Optimization of Equation (10)", fontsize = 12)
plt.xlabel("x", fontsize = 12)
plt.ylabel("y", fontsize = 12)

plt.colorbar(label = "Step Index")
plt.grid()

# plt.savefig('Figures\\Simulated Annealing Optimization of Equation (10).pdf')
plt.show()

"Simulated annealing to minimize equation (12)"
t = 0
x = 2.0
y = 2.0

path_x = []
path_y = []
path_g = []

t_index = T_0
while t_index > T_MIN:
    path_x.append(x)
    path_y.append(y)
    path_g.append(g(x, y))

    # Cooling
    t += 1
    t_index = T_0 * np.exp(-t / TAU)

    # Propose Monte Carlo step
    dx = gaussian()
    dy = gaussian()
    x_new = x + dx
    y_new = y + dy

    # Reject step if outside domain
    if not (0 < x_new < 50 and -20 < y_new < 20):
        continue

    # Energy difference
    dE = g(x_new, y_new) - g(x, y)

    # Metropolis probability
    if dE < 0 or np.random.rand() < np.exp(-dE / t_index):
        x, y = x_new, y_new

print(f"Final position: x = {x}, y = {y}")

"Plotting Simulated Annealing Optimization of Equation (12)"
plt.figure()

# Plotting simulated annealing optimization of Equation (12)
plt.scatter(path_x, path_y, c = np.arange(len(path_x)), cmap = 'viridis', s = 4)

# Labels
plt.title("Simulated Annealing Optimization of Equation (12)", fontsize = 12)
plt.xlabel("x", fontsize = 12)
plt.ylabel("y", fontsize = 12)

plt.colorbar(label = "Step Index")
plt.grid()

# plt.savefig('Figures\\Simulated Annealing Optimization of Equation (12).pdf')
plt.show()
