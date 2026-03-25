"""
Cart-pole co-design, open-loop vs closed-loop, both solved by Dymos
"""

from run_closedloop import main as main_cl
from run_openloop import main as main_ol
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # closed-loop CCD
    p1, _ = main_cl(flag_opt_design=True)

    # open-loop CCD
    p2, _ = main_ol(flag_opt_design=True)

    # plot
    time1 = p1.get_val('traj.phase.timeseries.time', units='s')
    x1 = p1.get_val('traj.phase.timeseries.states:x', units='m')
    theta1 = p1.get_val('traj.phase.timeseries.states:theta', units='rad')
    cost1 = p1.get_val('traj.phase.timeseries.states:cost')

    time2 = p2.get_val('traj.phase.timeseries.time', units='s')
    x2 = p2.get_val('traj.phase.timeseries.states:x', units='m')
    theta2 = p2.get_val('traj.phase.timeseries.states:theta', units='rad')
    cost2 = p2.get_val('traj.phase.timeseries.states:cost') 

    fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    axs[0].plot(time1, x1, color='k', alpha=0.5, label='Closed-loop')
    axs[0].plot(time2, x2, color='k', alpha=1.0, label='Open-loop')
    axs[0].set_ylabel('x, m')
    axs[0].legend()

    axs[1].plot(time1, theta1, color='b', alpha=0.5)
    axs[1].plot(time2, theta2, color='b', alpha=1.0)
    axs[1].set_ylabel('theta, rad')
    
    axs[2].plot(time1, cost1, color='r', alpha=0.5)
    axs[2].plot(time2, cost2, color='r', alpha=1.0)
    axs[2].set_ylabel('cost')
    axs[2].set_xlabel('time, s')
    axs[0].set_title('CCD - open-loop vs closed-loop')

    plt.savefig('dymos-cartpole-closed-vs-open.pdf', bbox_inches='tight')

