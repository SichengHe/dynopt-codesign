# plot comparison of two trajectories

import numpy as np
import pickle
import matplotlib.pyplot as plt

### colors = ['black', 'C0', 'C1', 'C2', 'C3']
colors = ['C0', 'C1', 'C2', 'C3']

def _plot_timehis(ax, results, yname, ylabel, y_scaler=1.0, set_xlabel=True, xvar='time'):
    ncase = len(results)

    for i in range(ncase):
        x = results[i][xvar]
        x -= x[0]
        ax.plot(x, results[i][yname] * y_scaler, color=colors[i])
    # END FOR

    ax.set_ylabel(ylabel)
    if set_xlabel:
        if xvar == 'time':
            xlabel = 'time [s]'
        elif xvar == 'x':
            xlabel = 'x [m]'
        ax.set_xlabel(xlabel)
    else:
        # turn off xtick labels
        ax.set_xticklabels([])
    

def plot_comparison(results, labels, phase_name=None):

    if phase_name is None:
        phase_name = ''
    else:
        phase_name = phase_name + '-'

    # number of cases to be plotted
    ncase = len(results)
    
    # --- xy trajectory ---
    plt.figure(figsize=(8, 3))
    for i in range(ncase):
        plt.plot(results[i]['x'] - results[i]['x'][0], results[i]['y'], label=labels[i], color=colors[i])
    plt.xlabel('x [m]')
    plt.ylabel('altitude [m]')
    plt.legend()
    plt.axis('equal')
    plt.savefig(phase_name + 'xy.pdf', bbox_inches='tight')

    # --- speed history ---
    fig, axs = plt.subplots(2)
    _plot_timehis(axs[0], results, 'vx', 'horizontal speed [m/s]', set_xlabel=False)
    _plot_timehis(axs[1], results, 'vy', 'vertical speed [m/s]')
    plt.savefig(phase_name + 'speed-time.pdf', bbox_inches='tight')

    fig, axs = plt.subplots(2)
    _plot_timehis(axs[0], results, 'vx', 'horizontal speed [m/s]', set_xlabel=False, xvar='x')
    _plot_timehis(axs[1], results, 'vy', 'vertical speed [m/s]', xvar='x')
    plt.savefig(phase_name + 'speed-loc.pdf', bbox_inches='tight')

    # --- control history ---
    fig, axs = plt.subplots(3)
    _plot_timehis(axs[0], results, 'theta', 'body pitch [deg]', y_scaler=180 / np.pi, set_xlabel=False)
    _plot_timehis(axs[1], results, 'thrust', 'horiz. thrust [N]', set_xlabel=False)
    _plot_timehis(axs[2], results, 'thrust_vert', 'vert. thrust [N]')
    plt.savefig(phase_name + 'theta-and-thrust-time.pdf', bbox_inches='tight')

    fig, axs = plt.subplots(3)
    _plot_timehis(axs[0], results, 'theta', 'body pitch [deg]', y_scaler=180 / np.pi, set_xlabel=False, xvar='x')
    _plot_timehis(axs[1], results, 'thrust', 'horiz. thrust [N]', set_xlabel=False, xvar='x')
    _plot_timehis(axs[2], results, 'thrust_vert', 'vert. thrust [N]', xvar='x')
    plt.savefig(phase_name + 'theta-and-thrust-loc.pdf', bbox_inches='tight')

    fig, axs = plt.subplots(2)
    # _plot_timehis(axs[0], res1, res2, 'theta', 'body tilt [deg]', y_scaler=180 / np.pi, set_xlabel=False)
    _plot_timehis(axs[0], results, 'omega', 'pusher RPM', set_xlabel=False, y_scaler=60 / 2 / np.pi)
    _plot_timehis(axs[1], results, 'omega_vert', 'lifter RPM', y_scaler=60 / 2 / np.pi)
    plt.savefig(phase_name + 'rotorRPM-time.pdf', bbox_inches='tight')

    fig, axs = plt.subplots(2)
    # _plot_timehis(axs[0], res1, res2, 'theta', 'body tilt [deg]', y_scaler=180 / np.pi, set_xlabel=False)
    _plot_timehis(axs[0], results, 'omega', 'pusher RPM', set_xlabel=False, y_scaler=60 / 2 / np.pi, xvar='x')
    _plot_timehis(axs[1], results, 'omega_vert', 'lifter RPM', y_scaler=60 / 2 / np.pi, xvar='x')
    plt.savefig(phase_name + 'rotorRPM-loc.pdf', bbox_inches='tight')

    """
    # --- some thrust analysis ---
    time1 = res1['time']
    time2 = res2['time']
    thrust_pusher_1 = res1['thrust']
    thrust_lifter_1 = res1['thrust_vert']
    thrust_pusher_2 = res2['thrust']
    thrust_lifter_2 = res2['thrust_vert']
    thrust_ratio_1 = thrust_lifter_1 / thrust_pusher_1
    thrust_ratio_2 = thrust_lifter_2 / thrust_pusher_2
    thrust_dir_1 = np.arctan2(thrust_lifter_1, thrust_pusher_1) + res1['theta']
    thrust_dir_2 = np.arctan2(thrust_lifter_2, thrust_pusher_2) + res2['theta']
    
    fig, axs = plt.subplots(2)
    axs[0].plot(time1, 1 / thrust_ratio_1, label=res1_label)
    axs[0].plot(time2, 1 / thrust_ratio_2, label=res2_label)
    axs[0].set_ylabel('Thrust ratio')
    axs[0].set_xticklabels([])

    axs[1].plot(time1, thrust_dir_1 * 180 / np.pi, label=res1_label)
    axs[1].plot(time2, thrust_dir_2 * 180 / np.pi, label=res2_label)
    axs[1].set_ylabel('Thrust direction [deg]')
    axs[1].set_xlabel('time [s]')
    plt.savefig(phase_name + 'thrust-details-time.pdf', bbox_inches='tight')
    """

    # --- power history ---
    # power sum
    fig, ax = plt.subplots(1)
    _plot_timehis(ax, results, 'power', 'Power [W]')
    plt.savefig(phase_name + 'power-time.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1)
    _plot_timehis(ax, results, 'power', 'Power [W]', xvar='x')
    plt.savefig(phase_name + 'power-loc.pdf', bbox_inches='tight')

    # lifter power
    fig, ax = plt.subplots(1, figsize=(5, 2))
    _plot_timehis(ax, results, 'power_for_lift', 'Power [W]')
    plt.savefig(phase_name + 'power-lifter-time.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1, figsize=(5, 2))
    _plot_timehis(ax, results, 'power_for_lift', 'Power [W]', xvar='x')
    plt.savefig(phase_name + 'power-lifter-loc.pdf', bbox_inches='tight')

    # cruiser power
    fig, ax = plt.subplots(1, figsize=(5, 2))
    _plot_timehis(ax, results, 'power_for_pusher', 'Power [W]')
    plt.savefig(phase_name + 'power-pusher-time.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1, figsize=(5, 2))
    _plot_timehis(ax, results, 'power_for_pusher', 'Power [W]', xvar='x')
    plt.savefig(phase_name + 'power-pusher-loc.pdf', bbox_inches='tight')

    # --- CL history ---
    fig, ax = plt.subplots(1)
    _plot_timehis(ax, results, 'CL_wing', 'CL')
    plt.savefig(phase_name + 'CL-time.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1)
    _plot_timehis(ax, results, 'CL_wing', 'CL', xvar='x')
    plt.savefig(phase_name + 'CL-loc.pdf', bbox_inches='tight')

    # --- rotor design ---
    # rotor 1 (pusher)
    nradii = np.size(results[0]['rotor1']['radii'])
    fig, axs = plt.subplots(2)
    for i in range(ncase):
        axs[0].plot(results[i]['rotor1']['radii'].reshape(nradii), results[i]['rotor1']['chord'].reshape(nradii), label=labels[i], color=colors[i])
        axs[1].plot(results[i]['rotor1']['radii'].reshape(nradii), results[i]['rotor1']['theta'].reshape(nradii), label=labels[i], color=colors[i])
    axs[0].set_ylabel('chord [m]')
    axs[0].legend()
    axs[0].set_xlim([0, np.max(results[0]['rotor1']['radii'][-1])])
    axs[0].set_xticklabels([])
    axs[1].set_xlabel('radius [m]')
    axs[1].set_ylabel('twist [deg]')
    axs[1].set_xlim([0, np.max(results[0]['rotor1']['radii'][-1])])
    ### axs[0].set_title('rotor (pusher) design')
    plt.savefig('rotor-pusher-design.pdf', bbox_inches='tight')

    # rotor 2 (vertical)
    nradii = np.size(results[0]['rotor2']['radii'])
    fig, axs = plt.subplots(2)
    for i in range(ncase):
        axs[0].plot(results[i]['rotor2']['radii'].reshape(nradii), results[i]['rotor2']['chord'].reshape(nradii), label=labels[i], color=colors[i])
        axs[1].plot(results[i]['rotor2']['radii'].reshape(nradii), results[i]['rotor2']['theta'].reshape(nradii), label=labels[i], color=colors[i])

    axs[0].set_ylabel('chord [m]')
    axs[0].legend()
    axs[0].set_xlim([0, np.max(results[0]['rotor2']['radii'][-1])])
    axs[0].set_xticklabels([])
    axs[1].set_xlabel('radius [m]')
    axs[1].set_ylabel('twist [deg]')
    axs[1].set_xlim([0, np.max(results[0]['rotor2']['radii'][-1])])
    ### axs[0].set_title('rotor (lifter) design')
    plt.savefig('rotor-lifter-design.pdf', bbox_inches='tight')

    # plt.show()

def print_energy_comparison(results, labels, phase_name, cruise_ranges):
    # print energy reduction of each phase
    ncase = len(results)
    # energy
    e = np.zeros(ncase)
    for i in range(ncase):
        e[i] = results[i]['energy'][-1] / 1000   # kWs

    # energy reduction (%)
    eref = e[0]
    if phase == 'cruise':
        # adjust reference energy for the given range
        eref = e[0] * np.array(cruise_ranges) / cruise_ranges[0]
    reduc = (eref - e) / eref * 100

    print('--- ' + phase_name + ' energy [kWs] ---')
    print(labels[0], ':', e[0])
    for i in range(1, ncase):
        print(labels[i], ':', e[i], '(-' + str(reduc[i]) + '%)')

def print_cruise_solution(results, labels):
    ncase = len(results)
    print('--- cruise solution ---')
    print(labels)
    keys = ['omega', 'omega_vert', 'thrust', 'thrust_vert', 'theta', 'CL']
    for key in keys:
        string = key + ':  '
        for i in range(ncase):
            string += str(results[i][key][0]) + ',  '
        # END FOR
        print(string)
    # END FOR

def plot_full_traj(res_climb, res_descent, cruise_range, yscaler):
    x1 = res_climb['x']
    y1 = res_climb['y']
    x2 = res_descent['x']
    y2 = res_descent['y']
    # adjust descent x
    x2 = x2 - x2[0] + x1[-1] + cruise_range

    x = np.concatenate((x1, [x1[-1], x1[-1] + cruise_range], x2))
    y = np.concatenate((y1, [y1[-1], y2[0]], y2))

    plt.figure(figsize=(8, 3))
    plt.plot(x, y * yscaler)
    plt.xlabel('x [m]')
    plt.ylabel('Altitude [m]')
    plt.axis('equal')
    plt.savefig('traj-full.pdf', bbox_inches='tight', transparent=True)
    # plt.show()


if __name__ == '__main__':
    # --- climb ---
    phases = ['climb', 'cruise', 'descent']

    path = '/Users/shugo/rsrc/UAV_traj_design/runscripts/results-1017/'
    file_baseline = 'naive-baseline/results_full_fixed_design.pkl'
    file_opt = 'co-design/results_full_coupled_design.pkl'

    # path = '/Users/shugo/rsrc/UAV_traj_design/results-LC-20km/'
    # file_baseline = 'result-0526-fixed-design/results_full_fixed_design.pkl'
    # file_opt = 'result-0526-desopt-case01/results_full_coupled_design.pkl'

    res_baseline_all = {}

    for phase in phases:
        # baseline fixed-design result
        with open(path + file_baseline, mode='rb') as file:
            res_all = pickle.load(file)
            res1 = res_all[phase]

            # plot reference trajectory
            ### plot_full_traj(res_all['climb'], res_all['descent'], 1000, yscaler=3)
        
        # coupled result 1
        with open(path + file_opt, mode='rb') as file:
            res_all = pickle.load(file)
            res2 = res_all[phase]

        """
        # --- one more results ---
        with open(path + 'var-design-2km-NODELTA/results_full_coupled_design.pkl', mode='rb') as file:
            res_all = pickle.load(file)
            res3 = res_all[phase]

        print_energy_comparison([res1, res3, res2], ['Sequential', 'Coupled (2km)', 'Coupled (10km)'], phase, [9400, 1400, 9400])

        if phase in ['climb', 'descent']:
            # plot
            plot_comparison(results=[res1, res3, res2], labels=['Sequential (baseline)', 'Coupled, 2 km mission', 'Coupled, 10 km mission'], phase_name=phase)
        else:
            # print cruise info
            print_cruise_solution(results=[res1, res3, res2], labels=['Sequential', 'Coupled (2km)', 'Coupled (10km)'])

        """
        # --- just two results ---
        print_energy_comparison([res1, res2], ['Baseline', 'Coupled'], phase, [9400, 9400])

        if phase in ['climb', 'descent']:
            # plot
            plot_comparison(results=[res1, res2], labels=['Baseline', 'Coupled'], phase_name=phase)
        else:
            # print cruise info
            print_cruise_solution(results=[res1, res2], labels=['Baseline', 'Coupled'])