
from slip.slip2d import slip2d
import numpy as np
import matplotlib.pyplot as plt
import math

aoa = 0.40586
rest_length = 0.27
stiffness = 2000
desired_height = 0.32
desired_xdot = 1
K_p = 0.1
slip = slip2d([0, desired_height, desired_xdot, 0, 0, 0], aoa, rest_length, 1/240, stiffness)

last_time = 0
sol = slip.step_apex_to_apex()
slip.set_target_vel(desired_xdot)

fig1, ax1 = plt.subplots()
ax1.plot(sol.y[0],sol.y[1])
fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1) # two axes on figure
ax2.plot(sol.t, sol.y[2])
ax3.plot(sol.t, sol.y[3])

fig1.canvas.manager.window.move(0, 0)
fig2.canvas.manager.window.move(710, 0)

for i in range(1,100):

    # Update time variables
    sol.t = last_time + sol.t
    last_time = sol.t[-1]
    # Plot Results
    ax1.plot(sol.y[0],sol.y[1])
    ax1.plot([sol.y[0][0], sol.y[4][0]],[sol.y[1][0], sol.y[5][0]],color="black")
    ax2.plot(sol.t, sol.y[2])
    ax3.plot(sol.t, sol.y[3])
    # Check if failed
    if sol.failed:
        break
    ## Use Raiberts controller
    #xdot_avg = sol.y[2][-1]#sum(sol.y[2]) / len(sol.y[2])
    #total_stance_time = sol.t_events[3][0] - sol.t_events[1][0]
    #x_f = xdot_avg*total_stance_time/2 + K_p * (sol.y[2][-1]-desired_xdot)
    #taoa = math.asin(x_f/rest_length)
    slip.set_aoa(aoa)
    # Update new state variables
    slip.update_state(sol.y[0][-1],sol.y[1][-1],sol.y[2][-1],sol.y[3][-1])
    # Print required variable
    #print("AOA: ",aoa/math.pi*180)
    #print("X_f: ",x_f)
    #print("total_stance_time: ",total_stance_time)
    ## Use slip model apex to apex
    sol = slip.step_apex_to_apex()

print(i," slip models could be solved")
print(last_time, "s is taken.")


ax1.set_title("X-Y graph")
ax1.set_xlabel("ground")
ax2.set_title("time vs xdot")
ax2.set_xlabel("time")
ax3.set_title("time vs ydot")
ax3.set_xlabel("time")

plt.show()