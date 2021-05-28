
from slip2d import slip2d
import numpy as np
import matplotlib.pyplot as plt

slip = slip2d([0, 0.30, 0, 0, 0, 0], 0, 0.29, 1/240)

last_time = 0

for i in range(1,10):

    sol = slip.step_apex_to_apex()
    sol.t = last_time + sol.t
    last_time = sol.t[-1]

    plt.plot(sol.t,sol.y[1], color='orange')
    slip.update_state(sol.y[0][-1],sol.y[1][-1],sol.y[2][-1],sol.y[3][-1])

plt.show()