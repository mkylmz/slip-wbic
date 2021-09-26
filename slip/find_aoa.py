
from slip.slip2d import slip2d
import numpy as np
import matplotlib.pyplot as plt
import math

SLIP_DT = 0.002
START_HEIGHT = 0.33
DESIRED_HEIGHT = 0.33
SLIP_KP = 0.01
SLIP_DESIRED_XDOT = 1
SLIP_AOA = 0.1519010943554561
SLIP_REST_LENGTH = 0.31
SLIP_STIFFNESS = 15000
SLIP_ACTIVATE_TIME = 5

slip = slip2d([0, DESIRED_HEIGHT, SLIP_DESIRED_XDOT, 0, 0, 0], SLIP_AOA, SLIP_REST_LENGTH, SLIP_DT, SLIP_STIFFNESS)

#K_p = 0.1
last_time = 0
sol = slip.step_apex_to_apex()
slip.set_target_vel(SLIP_DESIRED_XDOT)

fig1, ax1 = plt.subplots()
fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1) # two axes on figure

fig1.canvas.manager.window.move(0, 0)
fig2.canvas.manager.window.move(710, 0)

i = 0
while ( ( abs(DESIRED_HEIGHT-sol.y[1][-1]) > 10e-6) and i < 1000 ):
    i = i + 1

    # Check if failed
    if sol.failed:
        break

    height_diff = DESIRED_HEIGHT-sol.y[1][-1]
    if height_diff > 0:
        SLIP_AOA = SLIP_AOA * 1.001
    else:
        SLIP_AOA = SLIP_AOA * 0.999

    slip.set_aoa(SLIP_AOA)
    # Update new state variables
    slip.update_state( 0,  DESIRED_HEIGHT, SLIP_DESIRED_XDOT,  0)
    
    sol = slip.step_apex_to_apex()
    print("Slip models height: ", sol.y[1][-1])

print(i," slip models were tried")
print("Found SLIP_AOA as: ", SLIP_AOA)

# Plot Results
ax1.plot(sol.y[0],sol.y[1])
ax1.plot([sol.y[0][0], sol.y[4][0]],[sol.y[1][0], sol.y[5][0]],color="black")
ax2.plot(sol.t, sol.y[2])
ax3.plot(sol.t, sol.y[3])

ax1.set_title("X-Y graph")
ax1.set_xlabel("ground")
ax2.set_title("time vs xdot")
ax2.set_xlabel("time")
ax3.set_title("time vs ydot")
ax3.set_xlabel("time")

plt.show()