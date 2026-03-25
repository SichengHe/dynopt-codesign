import autograd.numpy as np
from autograd import jacobian, grad, elementwise_grad
import residual



"""
    Reduced residual class.
"""

class residual_reduced:

    def __init__(self, res_obj: residual.residual, w_fixed, q_fixed, res_reduced_ind:list, w_fixed_ind:list, q_fixed_ind:list):

        self.res_obj = res_obj

        self.w_fixed = w_fixed
        self.q_fixed = q_fixed

        self.res_reduced_ind = res_reduced_ind
        self.w_fixed_ind = w_fixed_ind
        self.q_fixed_ind = q_fixed_ind

        self.nstate = res_obj.get_state_size()
        self.nctrl = res_obj.get_ctrl_size()

        self.nstate_fixed = len(w_fixed_ind)
        self.nctrl_fixed = len(q_fixed_ind)

        self.nres_reduced = len(res_reduced_ind) 

    def map_w_q_2_theta(self, w, q):

        nstate = self.nstate
        nctrl = self.nctrl

        nstate_fixed = self.nstate_fixed
        nctrl_fixed = self.nctrl_fixed

        w_fixed_ind = self.w_fixed_ind
        q_fixed_ind = self.q_fixed_ind

        nres_reduced = self.nres_reduced

        theta = np.zeros(nres_reduced)

        w_free_ind = list(set(range(nstate)) - set(w_fixed_ind))
        w_free_ind.sort()
        q_free_ind = list(set(range(nctrl)) - set(q_fixed_ind))
        q_free_ind.sort()

        nstate_free = nstate - nstate_fixed
        nctrl_free = nctrl - nctrl_fixed

        # w contribution
        i_theta = 0
        for i in range(nstate_free):
            ind = w_free_ind[i]
            value = w[ind]

            theta[i_theta] = value

            i_theta += 1

        # q contribution
        for i in range(nctrl_free):
            ind = q_free_ind[i]
            value = q[ind]

            theta[i_theta] = value

            i_theta += 1

        return theta

    def map_theta_2_w_q(self, theta):

        nstate = self.nstate
        nctrl = self.nctrl
        
        w = np.zeros(nstate)
        q = np.zeros(nctrl)

        # Fill in the fixed
        w_fixed = self.w_fixed
        q_fixed = self.q_fixed

        w_fixed_ind = self.w_fixed_ind
        q_fixed_ind = self.q_fixed_ind

        nstate_fixed = self.nstate_fixed
        nctrl_fixed = self.nctrl_fixed

        for i in range(nstate_fixed):

            ind = w_fixed_ind[i]
            w[ind] = w_fixed[i]

        for i in range(nctrl_fixed):

            ind = q_fixed_ind(i)
            q[ind] = q_fixed[ind]

        # Fill in the free set by theta
        w_free_ind = list(set(range(nstate)) - set(w_fixed_ind))
        w_free_ind.sort()
        q_free_ind = list(set(range(nctrl)) - set(q_fixed_ind))
        q_free_ind.sort()

        nstate_free = nstate - nstate_fixed
        nctrl_free = nctrl - nctrl_fixed

        i_theta = 0
        for i in range(nstate_free):

            ind = w_free_ind[i]
            value = theta[i_theta]

            w[ind] = value

            i_theta += 1

        for i in range(nctrl_free):

            ind = q_free_ind[i]
            value = theta[i_theta]

            q[ind] = value

            i_theta += 1

        return w, q

    def map_res_2_res_reduced(self, res):

        nres_reduced = self.nres_reduced
        res_reduced_ind = self.res_reduced_ind

        res_reduced = np.zeros(nres_reduced)
        for i in range(nres_reduced):
            ind = res_reduced_ind[i]
            value = res[ind]

            res_reduced[i] = value

        return res_reduced

    def compute_org(self, theta, x, t):

        res_obj = self.res_obj
        w, q = self.map_theta_2_w_q(theta)

        res_val_org = res_obj.compute(w, q, x, t)

        return res_val_org

    def compute(self, theta, x, t):

        res_val_org = self.compute_org(theta, x, t)

        res_val = self.map_res_2_res_reduced(res_val_org)

        return res_val

    def compute_grad_state_org(self, theta, x, t):

        res_obj = self.res_obj
        w, q = self.map_theta_2_w_q(theta)

        p_res_p_w = res_obj.compute_grad_state(w, q, x, t)

        return p_res_p_w

    def compute_grad_ctrl_org(self, theta, x, t):

        res_obj = self.res_obj
        w, q = self.map_theta_2_w_q(theta)

        p_res_p_q = res_obj.compute_grad_ctrl(w, q, x, t)

        return p_res_p_q

    def compute_grad_state(self, theta, x, t):

        nstate = self.nstate
        nctrl = self.nctrl

        p_res_p_w = self.compute_grad_state_org(theta, x, t)
        p_res_p_q = self.compute_grad_ctrl_org(theta, x, t)

        # res -> res_reduced
        nres_reduced = self.nres_reduced
        p_res_reduced_p_w = np.zeros((nres_reduced, nstate))
        p_res_reduced_p_q = np.zeros((nres_reduced, nctrl))
        for i in range(nstate):
            p_res_reduced_p_w[:, i] = self.map_res_2_res_reduced(p_res_p_w[:, i])
        
        for i in range(nctrl):
            p_res_reduced_p_q[:, i] = self.map_res_2_res_reduced(p_res_p_q[:, i])

        
        # w, q -> theta
        nres_reduced = self.nres_reduced
        p_res_reduced_p_theta = np.zeros((nres_reduced, nres_reduced))
        for i in range(nres_reduced):
            p_res_reduced_i_p_w = p_res_reduced_p_w[i, :]
            p_res_reduced_i_p_q = p_res_reduced_p_q[i, :]

            p_res_reduced_i_p_theta = self.map_w_q_2_theta(p_res_reduced_i_p_w, p_res_reduced_i_p_q)

            p_res_reduced_p_theta[i, :] = p_res_reduced_i_p_theta[:]

        return p_res_reduced_p_theta

    def compute_grad_design_org(self, theta, x, t):

        res_obj = self.res_obj
        w, q = self.map_theta_2_w_q(theta)

        p_res_p_x = res_obj.compute_grad_design(w, q, x, t)

        return p_res_p_x

    def compute_grad_design(self, theta, x, t):

        nx = len(x)

        p_res_p_x = self.compute_grad_design_org(theta, x, t)

        # res -> res_reduced
        nres_reduced = self.nres_reduced

        p_res_reduced_p_x = np.zeros((nres_reduced, nx))

        for i in range(nx):
            p_res_reduced_p_x[:, i] = self.map_res_2_res_reduced(p_res_p_x[:, i])

        return p_res_reduced_p_x

    def get_state_size(self):

        nstate = self.nstate
        nctrl = self.nctrl

        nstate_fixed = self.nstate_fixed
        nctrl_fixed = self.nctrl_fixed

        return nstate + nctrl - nstate_fixed - nctrl_fixed








if __name__ == "__main__":
    
    def res_func(w, q, x, t):

        # Extract the parameters
        m = x[0]
        M = x[1]
        L = x[2]

        g = 10

        Sx = np.sin(w[2])
        Cx = np.cos(w[2])

        D1 = m * Cx**2 - (m + M)
        D2 = (m * Cx**2 - (m + M)) * L

        dx = np.array([w[1], 
        (1 / D1) * (-m * g * Sx * Cx - (q[0] + m * L * w[3]**2 * Sx)), 
        w[3], 
        (1 / D2) * ((m + M) * g * Sx + Cx * (q[0] + m * L * w[3]**2 * Sx))])

        return dx

    # Conduct AD
    p_res_p_w_func = jacobian(res_func, 0)
    p_res_p_q_func = jacobian(res_func, 1)
    p_res_p_x_func = jacobian(res_func, 2)

    # Construct the object
    nstate = 4
    nctrl = 1
    res = residual.residual(res_func, nstate, nctrl, p_res_p_w_func=p_res_p_w_func, p_res_p_q_func=p_res_p_q_func, p_res_p_x_func=p_res_p_x_func)

    w_fixed = np.array([1.0])
    q_fixed = np.array([])

    res_reduced_ind = range(4)
    w_fixed_ind = [0]
    q_fixed_ind = []

    res_reduced = residual_reduced(res, w_fixed, q_fixed, res_reduced_ind, w_fixed_ind, q_fixed_ind)

    # Test
    theta = np.random.rand(4)

    m = 2.0
    M = 7.5
    L = 1.0

    x = np.array([m, M, L])
    t = 0.

    res_val = res_reduced.compute(theta, x, t)
    d_res_d_theta_val = res_reduced.compute_grad_state(theta, x, t)
    d_res_d_x_val = res_reduced.compute_grad_design(theta, x, t)

    print("res_val", res_val)
    print("d_res_d_theta_val", d_res_d_theta_val)
    print("d_res_d_x_val", d_res_d_x_val)


