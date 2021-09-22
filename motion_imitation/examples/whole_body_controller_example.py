"""Example of whole body controller on A1 robot."""
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from absl import app
from absl import flags
from absl import logging
from datetime import datetime
import numpy as np
import os
import scipy.interpolate
import time
import math

import pybullet_data
from pybullet_utils import bullet_client
import pybullet  # pytype:disable=import-error

from mpc_controller import com_velocity_estimator, gait_generator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
#from mpc_controller import torque_stance_leg_controller
#import mpc_osqp
from mpc_controller import torque_stance_leg_controller_quadprog as torque_stance_leg_controller


from motion_imitation.robots import a1
from motion_imitation.robots import robot_config
from motion_imitation.robots.gamepad import gamepad_reader # type: ignore

from slip.slip2d import slip2d

flags.DEFINE_string("logdir", None, "where to log trajectories.")
flags.DEFINE_bool("use_gamepad", False,
                  "whether to use gamepad to provide control input.")
flags.DEFINE_bool("use_real_robot", False,
                  "whether to use real robot or simulation")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("plot_slip", False, "whether to plot slip results")
FLAGS = flags.FLAGS

_NUM_SIMULATION_ITERATION_STEPS = 300
_MAX_TIME_SECONDS = 10.

# Standing
# _DUTY_FACTOR = [1.] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0., 0., 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
# )

# Tripod
# _DUTY_FACTOR = [.8] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0.25, 0.5, 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.SWING,
# )

_STANCE_DURATION_SECONDS = [0.3] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).
# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)

#########################################################
############### MY SLIP BASED PARAMETERS ################
START_HEIGHT = 0.3
DESIRED_HEIGHT = 0.3
SLIP_KP = 0.01
SLIP_DESIRED_XDOT = 1
SLIP_AOA = 0.3667327599283879
SLIP_REST_LENGTH = 0.27
SLIP_STIFFNESS = 3000
SLIP_ACTIVATE_TIME = 5

def _generate_example_linear_angular_speed(t):
  """Creates an example speed profile based on time for demo purpose."""
  vx = 0.6
  vy = 0.2
  wz = 0.8

  time_points = (0, 5, 10, 15, 20, 25, 30)
  speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz),
                  (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="previous",
                                     fill_value="extrapolate",
                                     axis=0)(t)

  return speed[0:3], speed[3], False

def _generate_slip_trajectory_tracking(slip_sol, t, desired_speed, desired_height):
  if t < slip_sol.t[-1]:
    cur_timestep = t%slip_sol.t[-1]//0.001
    # state vector [ x, y, xdot, ydot, toe_x, toe_y]
    lin_vel = [slip_sol.y[2][int(cur_timestep)],0,0]
    ang_vel = 0
    body_height = slip_sol.y[1][int(cur_timestep)]
    finish_flag = False
  else:
    # state vector [ x, y, xdot, ydot, toe_x, toe_y]
    lin_vel = [desired_speed,0,0]
    ang_vel = 0
    body_height = desired_height
    finish_flag = True
  return lin_vel, ang_vel, body_height, finish_flag

def _setup_controller(robot):
  """Demonstrates how to create a locomotion controller."""
  desired_speed = (0, 0)
  desired_twisting_speed = 0

  gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor=_DUTY_FACTOR,
      initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
      initial_leg_state=_INIT_LEG_STATE)
  window_size = 20 if not FLAGS.use_real_robot else 1
  state_estimator = com_velocity_estimator.COMVelocityEstimator(
      robot, window_size=window_size)
  sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_height=robot.MPC_BODY_HEIGHT,
      foot_clearance=0.01)

  st_controller = torque_stance_leg_controller.TorqueStanceLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_body_height=robot.MPC_BODY_HEIGHT
      ,#qp_solver = mpc_osqp.QPOASES #or mpc_osqp.OSQP
      )

  controller = locomotion_controller.LocomotionController(
      robot=robot,
      gait_generator=gait_generator,
      state_estimator=state_estimator,
      swing_leg_controller=sw_controller,
      stance_leg_controller=st_controller,
      clock=robot.GetTimeSinceReset)
  return controller


def _update_controller_params(controller, lin_speed, ang_speed):
  controller.swing_leg_controller.desired_speed = lin_speed
  controller.swing_leg_controller.desired_twisting_speed = ang_speed
  controller.stance_leg_controller.desired_speed = lin_speed
  controller.stance_leg_controller.desired_twisting_speed = ang_speed

def _update_controller_params_slip(controller, lin_speed, ang_speed, body_height):

  controller.swing_leg_controller.desired_speed = lin_speed
  controller.swing_leg_controller.desired_twisting_speed = ang_speed
  controller.swing_leg_controller._desired_height = np.array((0, 0, body_height - 0.01))
  controller.stance_leg_controller.desired_speed = lin_speed
  controller.stance_leg_controller.desired_twisting_speed = ang_speed
  controller.stance_leg_controller._desired_body_height = body_height

def check_apex(controller, old_z_vel):
  robot_z_vel = controller.state_estimator._com_velocity_world_frame[2]
  if old_z_vel >= 0 and robot_z_vel <= 0:
    return True
  return False

def main(argv):
  """Runs the locomotion controller example."""
  del argv # unused

  # Construct simulator
  if FLAGS.show_gui and not FLAGS.use_real_robot:
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
  else:
    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
  p.setPhysicsEngineParameter(numSolverIterations=30)
  p.setTimeStep(0.001)
  p.setGravity(0, 0, -9.8)
  p.setPhysicsEngineParameter(enableConeFriction=0)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  p.loadURDF("plane.urdf")

  # Construct robot class:
  if FLAGS.use_real_robot:
    from motion_imitation.robots import a1_robot
    robot = a1_robot.A1Robot(
        pybullet_client=p,
        motor_control_mode=robot_config.MotorControlMode.HYBRID,
        enable_action_interpolation=False,
        time_step=0.002,
        action_repeat=1)
  else:
    robot = a1.A1(p,
                  motor_control_mode=robot_config.MotorControlMode.HYBRID,
                  enable_action_interpolation=False,
                  reset_time=2,
                  time_step=0.002,
                  action_repeat=1)

  controller = _setup_controller(robot)

  controller.reset()
  """
  if FLAGS.use_gamepad:
    gamepad = gamepad_reader.Gamepad()
    command_function = gamepad.get_command
  else:
    command_function = _generate_example_linear_angular_speed
  """

  aoa = SLIP_AOA
  rest_length = SLIP_REST_LENGTH
  desired_height = DESIRED_HEIGHT
  dt = 0.001
  myslip = slip2d([0, desired_height, 0, 0, 0, 0], aoa, rest_length, dt, SLIP_STIFFNESS)
  command_function = _generate_slip_trajectory_tracking

  if FLAGS.logdir:
    logdir = os.path.join(FLAGS.logdir,
                          datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(logdir)

  start_time = robot.GetTimeSinceReset()
  current_time = start_time
  com_vels, imu_rates, actions = [], [], []
  old_z_vel = 0
  K_p = SLIP_KP
  xdot_des = SLIP_DESIRED_XDOT
  total_stance_time = 0
  total_flight_time = 0
  total_motion_time = 0
  slip_active = False
  slip_solved = False
  finish_flag = False

  if FLAGS.plot_slip:
    slip_time = np.array([])
    slip_sols = np.array([[],[],[],[],[],[]])
    last_time = 0
    last_x = 0
    robot_cur_time = 0
    robot_times = np.array([])
    robot_pos_x = np.array([])
    robot_pos_z = np.array([])
    robot_vel_x = np.array([])
    robot_vel_z = np.array([])

  while current_time - start_time < _MAX_TIME_SECONDS:
    #time.sleep(0.0008) #on some fast computer, works better with sleep on real A1?
    start_time_robot = current_time
    start_time_wall = time.time()

    ## check whether to start slip
    if not finish_flag and (current_time - start_time > SLIP_ACTIVATE_TIME):
        slip_active = True


    ## Apex-to-Apex slip model/controller
    if ( slip_active and check_apex(controller,old_z_vel) ):

      if slip_solved:
        if slip_current_time > slip_sol.t[-1]:
          finish_flag = True
          slip_active = False
      else:
        # Get new state variables
        robot_vel = controller.state_estimator._com_velocity_body_frame
        robot_height = controller.stance_leg_controller._robot_com_position[2]
        #robot_height = controller._robot.GetRobotPosition()[2]
        ## Use Raiberts controller
        """xdot_avg = robot_vel[0]
        x_f = xdot_avg*total_stance_time/2 + K_p * (robot_vel[0]-xdot_des)
        if (x_f > rest_length):
            x_f = rest_length
        aoa = math.asin(x_f/rest_length)"""
        myslip.set_aoa(SLIP_AOA)
        myslip.set_target_vel(SLIP_DESIRED_XDOT)
        # Update slip state
        myslip.update_state( 0, robot_height, robot_vel[0], robot_vel[2])
        ## Solve slip model
        slip_sol = myslip.step_apex_to_apex()
        if slip_sol.failed:
          slip_solved = False
          print("Slip failed")
        else:
          total_motion_time = slip_sol.t_events[5][0]
          total_flight_time = slip_sol.t_events[1][0] + slip_sol.t_events[5][0] - slip_sol.t_events[3][0]
          total_stance_time = (total_motion_time-total_flight_time)
          controller.gait_generator.change_gait_parameters([total_stance_time]*4,[total_stance_time/total_motion_time]*4)
          slip_solved = True
          #current_time = 0
          slip_current_time = 0

          if FLAGS.plot_slip:
            # Update variables
            slip_time = np.concatenate([slip_time, (current_time + slip_sol.t) ])
            #last_time = slip_time[-1]
            last_x = controller._robot.GetRobotPosition()[0]
            slip_sol.y[0] = last_x + slip_sol.y[0]
            slip_sols = np.concatenate([slip_sols, slip_sol.y ],axis=1)

    ## Update old z velocity
    old_z_vel = controller.state_estimator._com_velocity_world_frame[2]

    # Updates the controller behavior parameters.
    if slip_active and slip_solved:
      lin_speed, ang_speed, body_height, finish_flag = command_function(slip_sol, slip_current_time, xdot_des, desired_height)
      _update_controller_params_slip(controller, lin_speed, ang_speed, body_height)
      slip_current_time += dt
      if FLAGS.plot_slip:
        robot_cur_time = current_time
        robot_times = np.append(robot_times,robot_cur_time)
        robot_vel = controller.state_estimator._com_velocity_world_frame
        robot_pos = controller._robot.GetRobotPosition()
        robot_pos_x = np.append(robot_pos_x, robot_pos[0])
        robot_pos_z = np.append(robot_pos_z, robot_pos[2])
        robot_vel_x = np.append(robot_vel_x, robot_vel[0])
        robot_vel_z = np.append(robot_vel_z, robot_vel[2])
    else:
      _update_controller_params_slip(controller, [SLIP_DESIRED_XDOT,0,0], 0, DESIRED_HEIGHT)
    controller.update()
    hybrid_action, _ = controller.get_action()
    com_vels.append(np.array(robot.GetBaseVelocity()).copy())
    imu_rates.append(np.array(robot.GetBaseRollPitchYawRate()).copy())
    actions.append(hybrid_action)
    robot.Step(hybrid_action)
    current_time = robot.GetTimeSinceReset()
    p.resetDebugVisualizerCamera(1.5, 30, -35, controller._robot.GetRobotPosition())

    if not FLAGS.use_real_robot:
      expected_duration = current_time - start_time_robot
      actual_duration = time.time() - start_time_wall
      if actual_duration < expected_duration:
        time.sleep(expected_duration - actual_duration)
    #print("actual_duration=", actual_duration)
  """if FLAGS.use_gamepad:
    gamepad.stop()"""

  if FLAGS.logdir:
    np.savez(os.path.join(logdir, 'action.npz'),
             action=actions,
             com_vels=com_vels,
             imu_rates=imu_rates)
    logging.info("logged to: {}".format(logdir))

  if FLAGS.plot_slip: 
    import matplotlib.pyplot as plt
    fig1, ax1 = plt.subplots()
    fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1) # two axes on figure
    fig3, ax4 = plt.subplots()
    fig1.canvas.manager.window.move(0, 0)
    fig2.canvas.manager.window.move(710, 0)
    fig3.canvas.manager.window.move(0, 585)

    # Plot Results
    ax1.plot(slip_sols[0],slip_sols[1],'.',markersize=1)
    ax1.plot(robot_pos_x,robot_pos_z)
    ax2.plot(slip_time, slip_sols[2],'.',markersize=1)
    ax2.plot(robot_times, robot_vel_x)
    ax3.plot(slip_time, slip_sols[3],'.',markersize=1)
    ax3.plot(robot_times, robot_vel_z)
    ax4.plot(slip_time, slip_sols[1],'.',markersize=1)
    ax4.plot(robot_times, robot_pos_z)

    ax1.set_title("X-Y graph")
    ax1.set_xlabel("ground")
    ax2.set_title("time vs xdot")
    ax2.set_xlabel("time")
    ax3.set_title("time vs ydot")
    ax3.set_xlabel("time")
    ax4.set_title("time vs y")
    ax4.set_xlabel("time")
    
    plt.show()


if __name__ == "__main__":
  app.run(main)
