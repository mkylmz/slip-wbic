
from slip.slip2d import slip2d
import numpy as np
import matplotlib.pyplot as plt
import math

aoa = 0.428357
rest_length = 0.27
stiffness = 3000
desired_height = 0.30
desired_xdot = 1
slip = slip2d([0, desired_height, desired_xdot, 0, 0, 0], aoa, rest_length, 1/240, stiffness)

#K_p = 0.1
last_time = 0
sol = slip.step_apex_to_apex()
slip.set_target_vel(desired_xdot)

fig1, ax1 = plt.subplots()
fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1) # two axes on figure

fig1.canvas.manager.window.move(0, 0)
fig2.canvas.manager.window.move(710, 0)

i = 0
while ( ( abs(desired_height-sol.y[1][-1]) > 10e-6) and i < 1000 ):
    i = i + 1

    # Check if failed
    if sol.failed:
        break

    height_diff = desired_height-sol.y[1][-1]
    if height_diff > 0:
        aoa = aoa * 1.001
    else:
        aoa = aoa * 0.999

    slip.set_aoa(aoa)
    # Update new state variables
    slip.update_state( 0,  desired_height, desired_xdot,  0)
    
    sol = slip.step_apex_to_apex()
    print("Slip models height: ", sol.y[1][-1])

print(i," slip models were tried")
print("Found aoa as: ", aoa)

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