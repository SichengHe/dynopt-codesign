import openmdao.api as om
import matplotlib.pyplot as plt

def get_solution(case):
    # get solution of some iteration

    # get time history of states, controls, and some other variables. Put into a dict
    traj = {}
    traj['time'] = case['traj.phase.timeseries.time']
    traj['x'] = case['traj.phase.timeseries.states:x']
    traj['y'] = case['traj.phase.timeseries.states:y']
    traj['vx'] = case['traj.phase.timeseries.states:vx']
    traj['vy'] = case['traj.phase.timeseries.states:vy']
    traj['theta'] = case['traj.phase.timeseries.states:theta']
    traj['theta_vel'] = case['traj.phase.timeseries.states:theta_vel']
    traj['cost'] = case['traj.phase.timeseries.states:cost']   # quadratic cost
    traj['power'] = case['traj.phase.timeseries.power']   # power consumption, sum of all rotors
    traj['thrust1'] = case['traj.phase.timeseries.thrust_vert_1']
    traj['thrust2'] = case['traj.phase.timeseries.thrust_vert_2']
    traj['omega1'] = case['traj.phase.timeseries.omega_vert_1']   # rotor speed
    traj['omega2'] = case['traj.phase.timeseries.omega_vert_2']

    # get rotor blade design
    design = {}
    design['chord_cp'] = case['rotor_vert|chord_cp']   # spline control point
    design['theta_cp'] = case['rotor_vert|theta_cp']   # spline control point
    design['radii'] = case['trim.dynamics.propulsion_vert_1.BEMT.radii']   # cell-center value of blade elements
    design['chord'] = case['trim.dynamics.propulsion_vert_1.BEMT.chord']   # cell-center value of blade elements
    design['theta'] = case['trim.dynamics.propulsion_vert_1.BEMT.theta']   # cell-center value of blade elements
    design['K'] = case['K']   # control matrix

    return traj, design


if __name__ == '__main__':
    file = '/Users/shugo/rsrc/2022-09-He-LQRCCD/code/dymos_quadrotor/runscripts/opt_log_closedloop.sql'
    cr = om.CaseReader(file)
    cases_name = cr.list_cases('driver')
    
    # number of iterations (function calls)
    num_iter = len(cases_name)

    # loop over iterations and get each solutions
    plt.figure()
    for i in range(num_iter):
        # get solution iteration i
        traj, design = get_solution(cr.get_case(cases_name[i]))

        # just plot x-y path of every iterations
        plt.plot(traj['x'], traj['y'])

    plt.show()






    