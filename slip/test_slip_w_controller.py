
from slip2d import slip2d
import numpy as np
import matplotlib.pyplot as plt
import math

rest_length = 0.27
slip = slip2d([0, 0.32, 0, 0, 0, 0], 0, rest_length, 1/240)

last_time = 0
sol = slip.step_apex_to_apex()
plt.plot(sol.y[0],sol.y[1], color='orange')
K_p = 0.01
xdot_des = 0.5
slip.set_target_vel(xdot_des)

for i in range(1,30):


    # Update time variables
    sol.t = last_time + sol.t
    last_time = sol.t[-1]
    ## Plot Results
    plt.plot(sol.y[0],sol.y[1], color='orange')
    if sol.failed:
        break
    # Update new state variables
    slip.update_state(sol.y[0][-1],sol.y[1][-1],sol.y[2][-1],sol.y[3][-1])
    ## Use Raiberts controller
    xdot_avg = sum(sol.y[2]) / len(sol.y[2])
    total_stance_time = sol.t_events[3][0] - sol.t_events[1][0]
    x_f = xdot_avg*total_stance_time/2 + K_p * (sol.y[2][-1]-xdot_des)
    if (x_f > rest_length):
        x_f = rest_length
    aoa = math.asin(x_f/rest_length)
    slip.set_aoa(aoa)
    ## Use slip model apex to apex
    sol = slip.step_apex_to_apex()

print(i," timesteps could be calculated")
plt.show()