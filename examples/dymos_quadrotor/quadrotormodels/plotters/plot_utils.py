import numpy as np
import matplotlib.pyplot as plt

def plot_multiphase(traj_name, phases, axes, title, figsize=(8, 4), p_sol=None, y_units=None, connect_phase=True):
    ### nrows = len(axes)
    nrows = 1

    # create new figure
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=figsize)
    fig.suptitle(title)

    if nrows == 1:
        axs = [axs]

    # loop over entries to be plotted
    for i, (x, y, xlabel, ylabel) in enumerate(axes):
        color = 'C' + str(i)

        # get data
        x_sol = []  # time series 
        y_sol = []
        x_ends = []  # init and end of each phase
        y_ends = []
        for phase in phases:
            if p_sol is not None:
                # optimization solution
                xname = traj_name + '.' + phase + '.timeseries.' + x
                yname = traj_name + '.' + phase + '.timeseries.' + y
                xval = p_sol.get_val(xname)
                yval = p_sol.get_val(yname, units=y_units)
                xval = list(xval.reshape(np.size(xval,)))  # convert to 1D list
                yval = list(yval.reshape(np.size(yval,)))

                # connect time series
                x_sol += xval
                y_sol += yval
                # end points
                x_ends.append(xval[0])
                x_ends.append(xval[-1])
                y_ends.append(yval[0])
                y_ends.append(yval[-1])

                if not(connect_phase):
                    # plot each phase separately (but on the same figure)
                    axs[0].plot(xval, yval, 'o-', ms=4, color=color, label=y)
        # END FOR

        # plot entire trajectory
        if connect_phase and p_sol is not None:
            if p_sol is not None:
                # solution time series
                axs[0].plot(x_sol, y_sol, 'o-', ms=4, color=color)

                # end points
                axs[0].plot(x_ends, y_ends, 's', ms=10, color=color)

        # set axis and title
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        fig.suptitle(title)
        ## fig.legend(loc='lower center', ncol=2)

    fig.legend()
    plt.grid()
    plt.savefig('./traj-plots/' + title + '.pdf', bbox_inches='tight')
    return fig, axs

def plot_traj_details(traj_name, phases, p_sol, title=None, figsize=(8, 2)):
    # plot body atitude and thrust vectors on top of the trajectory
    # TODO: generalize for different VTOL configuration

    # --- get solution ---
    x = []
    y = []
    theta = []
    thrust1 = []
    thrust2 = []

    for phase in phases:
        name = traj_name + '.' + phase
        xval = p_sol.get_val(name + '.timeseries.states:x', units='m')
        yval = p_sol.get_val(name + '.timeseries.states:y', units='m')
        theta_val = p_sol.get_val(name + '.timeseries.controls:theta', units='rad')
        thrust1_val = p_sol.get_val(name + '.timeseries.thrust', units='N')
        thrust2_val = p_sol.get_val(name + '.timeseries.thrust_vert', units='N')
        # convert to 1D and append
        ndim = np.size(xval)
        x += list(xval.reshape(ndim))
        y += list(yval.reshape(ndim))
        theta += list(theta_val.reshape(ndim))
        thrust1 += list(thrust1_val.reshape(ndim))
        thrust2 += list(thrust2_val.reshape(ndim))
    # END FOR

    # thrust magnitude and direction
    thrust1 = np.array(thrust1)
    thrust2 = np.array(thrust2)
    thrust = (thrust1**2 + thrust2**2)**0.5
    thrust_direc = np.arctan2(thrust2, thrust1) + theta
    # normalize thrust
    ### max_thrust = max(np.max(thrust1), np.max(thrust2))
    max_thrust = np.max(thrust)
    thrust1 /= max_thrust
    thrust2 /= max_thrust
    thrust /= max_thrust

    # prepare arrays for airfoil/thrust vector plot
    frequency = 5
    x_plt = []
    y_plt = []
    theta_plt = []
    thrust_plt = []
    thrust_direc_plt = []
    for i in range(len(x)):   # loop over node
        if i % frequency == 0 or i == len(x) - 1:
            x_plt.append(x[i])
            y_plt.append(y[i])
            theta_plt.append(theta[i])
            thrust_plt.append(thrust[i])
            thrust_direc_plt.append(thrust_direc[i])
    # END FOR

    # --- plot ---
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        
    # trajectory
    ax.plot(x, y, linewidth=1)

    # airfoil attitude
    _plot_traj_and_theta(ax, x_plt, y_plt, theta_plt)

    # thrust vector
    ### _plot_traj_and_thrust(ax, x, y, thrust1, theta)  # pusher
    ### _plot_traj_and_thrust(ax, x, y, thrust2, theta + np.pi/2)  # lifter
    _plot_traj_and_thrust(ax, x_plt, y_plt, thrust_plt, thrust_direc_plt)   # sum of pusher and lifter

    ax.set_xlabel('x location, m')
    ax.set_ylabel('altitude, m')
    plt.axis('equal')

    plt.savefig('./traj-plots/' + title + '.pdf', bbox_inches='tight')
    return None

def _plot_traj_and_theta(fig_ax, x, y, theta, scaler=30, profile='airfoil'):
    # x, y, theta: time histories
    # scaler: chord length scaler in the plot
    # frequency: frequency of plotting the airfoil

    if profile == 'airfoil':
        # --- prepare airfoil ---
        # unit-chord airfoil shape (t/c=0.12)
        n_af_pts = 50
        t = 0.12
        af_x = np.linspace(0, 1, n_af_pts)
        af_y = t/.2*(.296*np.sqrt(af_x)-.126*af_x-.3516*af_x**2+.2843*af_x**3-.1015*af_x**4)  # upper surface
        ## af_y2 = -af_y1[::-1]
        af_x = np.concatenate((af_x[:-1], af_x[::-1]))  # full airfoil, TE on right
        af_y = np.concatenate((af_y[:-1], -af_y[::-1]))
        af_xy = np.vstack((af_x, af_y)) * scaler   # scaling
        af_xy[0, :] -= scaler * 0.25  # shift center
        # rotate 180 deg so that TE is on the right
        rot_mat = np.array([[-1, 0], [0, -1]])
        af_xy = np.dot(rot_mat, af_xy)
    elif profile == 'plate':
        # just flat plate
        af_x = np.linspace(-0.5, 0.5, 3)
        af_y = np.zeros(3)
        af_xy = np.vstack((af_x, af_y)) * scaler

    # --- plot atitude ---
    for i in range(len(x)):   # loop over node
        # plot vehicle atitude (tilt angle)
        rot_mat = np.array([[np.cos(theta[i]), -np.sin(theta[i])], [np.sin(theta[i]), np.cos(theta[i])]])
        af_xy_rotated = np.dot(rot_mat, af_xy)
        
        fig_ax.plot(af_xy_rotated[0, :] + x[i], af_xy_rotated[1, :] + y[i], color='black', linewidth=0.5)
    # END FOR
    return None

def _plot_traj_and_thrust(ax, x, y, thrust, thrust_direc, color='red', scaler=50):
    # x, y, thrust, thrust_direc: time history
    # thrust magnitude should be normalized (e.g., max(thrust)=1)

    # --- plot thrust vector ---
    for i in range(len(x)):   # loop over node
        thrust_dx = thrust[i] * np.cos(thrust_direc[i]) * scaler
        thrust_dy = thrust[i] * np.sin(thrust_direc[i]) * scaler
        ax.arrow(x[i], y[i], thrust_dx, thrust_dy, width=scaler / 50, fc=color, ec=color)
    # END FOR
    return None


def plot_traj_details_multirotor(traj_name, phases, p_sol, title=None, figsize=(8, 2)):
    # plot body atitude and thrust vectors on top of the trajectory, for multirotor 3DoF model

    # --- get solution ---
    x = []
    y = []
    theta = []
    thrust1 = []
    thrust2 = []

    for phase in phases:
        name = traj_name + '.' + phase
        xval = p_sol.get_val(name + '.timeseries.states:x', units='m')
        yval = p_sol.get_val(name + '.timeseries.states:y', units='m')
        theta_val = p_sol.get_val(name + '.timeseries.states:theta', units='rad')
        thrust1_val = p_sol.get_val(name + '.timeseries.thrust_vert_1', units='N')
        thrust2_val = p_sol.get_val(name + '.timeseries.thrust_vert_2', units='N')
        # convert to 1D and append
        ndim = np.size(xval)
        x += list(xval.reshape(ndim))
        y += list(yval.reshape(ndim))
        theta += list(theta_val.reshape(ndim))
        thrust1 += list(thrust1_val.reshape(ndim))
        thrust2 += list(thrust2_val.reshape(ndim))
    # END FOR

    # normalize thrust
    max_thrust = max(np.max(thrust1), np.max(thrust2))
    thrust1 /= max_thrust
    thrust2 /= max_thrust

    # --- plot ---
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # first, plot XY path and tilt angle
    _plot_traj_and_theta(ax, x, y, theta, profile='plate', scaler=1)
    # next, plot thrust vectors.
    offset = 1 / 2
    offset_x = offset * np.cos(np.array(theta))
    offset_y = offset * np.sin(np.array(theta))
    
    _plot_traj_and_thrust(ax, np.array(x) + offset_x, np.array(y) + offset_y, thrust1, np.array(theta) + np.pi / 2, scaler=1)  # lifter 1
    _plot_traj_and_thrust(ax, np.array(x) - offset_x, np.array(y) - offset_y, thrust2, np.array(theta) + np.pi / 2, scaler=1)  # lifter 2

    ax.set_xlabel('x location, m')
    ax.set_ylabel('altitude, m')
    plt.axis('equal')

    plt.savefig('./traj-plots/' + title + '.pdf', bbox_inches='tight')
    return None


if __name__ == '__main__':
    ### _plot_traj_and_theta(0, 0, 0)

    # load data
    import pickle
    with open('init_guess_LC_BEMT.pkl', mode='rb') as file:  # TODO: update path to your local pkl file
        res = pickle.load(file)
        init_guess_climb = res['climb']
        init_guess_cruise = res['cruise']
        init_guess_descent = res['descent']
    
    # ------------------------------------
    # plot climb
    x = init_guess_climb['x']
    ndim = np.size(x)
    x = x.reshape(ndim)
    y = init_guess_climb['y'].reshape(ndim)
    theta = init_guess_climb['theta'].reshape(ndim)
    thrust1 = init_guess_climb['thrust'].reshape(ndim)
    thrust2 = init_guess_climb['thrust_vert'].reshape(ndim)
    
    # vector sum
    thrust = (thrust1**2 + thrust2**2)**0.5
    thrust_direc = np.arctan2(thrust2, thrust1) + theta
    # normalize thrust
    ### max_thrust = max(np.max(thrust1), np.max(thrust2))
    max_thrust = np.max(thrust)
    thrust1 /= max_thrust
    thrust2 /= max_thrust
    thrust /= max_thrust

    # plot atitude and thrust vectors
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 2))
    _plot_traj_and_theta(ax, x, y, theta)
    ### _plot_traj_and_thrust(ax, x, y, thrust1, theta)  # pusher
    ### _plot_traj_and_thrust(ax, x, y, thrust2, theta + np.pi/2)  # lifter
    _plot_traj_and_thrust(ax, x, y, thrust, thrust_direc)   # sum of pusher and lifter

    ax.set_xlabel('x location, m')
    ax.set_ylabel('altitude, m')
    plt.axis('equal')
    plt.title('climb')
    plt.savefig('climb.png', dpi=600, bbox_inches='tight')

    # ------------------------------------
    # plot descent
    x = init_guess_descent['x']
    ndim = np.size(x)
    x = x.reshape(ndim)
    y = init_guess_descent['y'].reshape(ndim)
    theta = init_guess_descent['theta'].reshape(ndim)
    thrust1 = init_guess_descent['thrust'].reshape(ndim)
    thrust2 = init_guess_descent['thrust_vert'].reshape(ndim)
    
    # vector sum
    thrust = (thrust1**2 + thrust2**2)**0.5
    thrust_direc = np.arctan2(thrust2, thrust1) + theta
    # normalize thrust
    ### max_thrust = max(np.max(thrust1), np.max(thrust2))
    ### max_thrust = np.max(thrust)   # use same normalization as climb
    thrust1 /= max_thrust
    thrust2 /= max_thrust
    thrust /= max_thrust

    # plot atitude and thrust vectors
    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 2))
    _plot_traj_and_theta(ax2, x, y, theta)
    ### _plot_traj_and_thrust(ax, x, y, thrust1, theta)  # pusher
    ### _plot_traj_and_thrust(ax, x, y, thrust2, theta + np.pi/2)  # lifter
    _plot_traj_and_thrust(ax2, x, y, thrust, thrust_direc)   # sum of pusher and lifter

    ax2.set_xlabel('x location, m')
    ax2.set_ylabel('altitude, m')
    plt.axis('equal')
    plt.title('descent')
    plt.savefig('descent.png', dpi=600, bbox_inches='tight')
    plt.show()
    