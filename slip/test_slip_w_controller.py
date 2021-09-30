
from slip.slip2d import slip2d
import numpy as np
import matplotlib.pyplot as plt
import math

SLIP_DT = 0.002
START_HEIGHT = 0.32
DESIRED_HEIGHT = 0.32
SLIP_KP = 0.01
SLIP_DESIRED_XDOT = 1
SLIP_AOA = 0.272251250523425
SLIP_REST_LENGTH = 0.30
SLIP_STIFFNESS = 5000
SLIP_ACTIVATE_TIME = 5

slip = slip2d([0, DESIRED_HEIGHT, SLIP_DESIRED_XDOT, 0, 0, 0], SLIP_AOA, SLIP_REST_LENGTH, SLIP_DT, SLIP_STIFFNESS)

#K_p = 0.1
last_time = 0
sol = slip.step_apex_to_apex()
slip.set_target_vel(SLIP_DESIRED_XDOT)

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
    #x_f = xdot_avg*total_stance_time/2 + K_p * (sol.y[2][-1]-SLIP_DESIRED_XDOT)
    #taoa = math.asin(x_f/SLIP_REST_LENGTH)
    slip.set_aoa(SLIP_AOA)
    # Update new state variables
    slip.update_state(sol.y[0][-1],sol.y[1][-1],sol.y[2][-1],sol.y[3][-1])
    # Print required variable
    #print("SLIP_AOA: ",SLIP_AOA/math.pi*180)
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