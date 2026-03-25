# import numpy as np
import autograd.numpy as np
from autograd import jacobian, grad, elementwise_grad

"""
    Residual class.
    For dynamical system of the form
        wdot = r(w, q, x) 

    # Inputs: 
    * w: state variable.
    * q: control variable.
    * x: design variable.
    * t: time. time in not an input for grad_J_* and grad_G_* because these linearization will be done at the steady-state (hence should be time-independent)
"""
class residual:
    def __init__(self, res_func, nstate, nctrl, p_res_p_w_func=None, p_res_p_q_func=None, p_res_p_x_func=None, p_J_p_w_func=None, p_J_p_q_func=None, p_J_p_x_func=None, p_G_p_w_func=None, p_G_p_q_func=None, p_G_p_x_func=None, reset_cache_func=None):

        self.res_func = res_func

        self.p_res_p_w_func = p_res_p_w_func
        self.p_res_p_q_func = p_res_p_q_func
        self.p_res_p_x_func = p_res_p_x_func

        self.p_J_p_w_func = p_J_p_w_func
        self.p_J_p_q_func = p_J_p_q_func
        self.p_J_p_x_func = p_J_p_x_func

        self.p_G_p_w_func = p_G_p_w_func
        self.p_G_p_q_func = p_G_p_q_func
        self.p_G_p_x_func = p_G_p_x_func

        # function to reset the analysis/derivatives cache
        self.reset_cache_func = reset_cache_func

        self.nstate = nstate
        self.nctrl = nctrl

    def compute(self, w, q, x, t):

        return self.res_func(w, q, x, t)
    
    def compute_grad_state(self, w, q, x, t):

        return self.p_res_p_w_func(w, q, x, t)

    def compute_grad_J_state(self, w, q, x):

        return self.p_J_p_w_func(w, q, x)

    def compute_grad_J_ctrl(self, w, q, x):

        return self.p_J_p_q_func(w, q, x)

    def compute_grad_J_design(self, w, q, x):

        return self.p_J_p_x_func(w, q, x)

    def compute_grad_G_state(self, w, q, x):
    
        return self.p_G_p_w_func(w, q, x)

    def compute_grad_G_ctrl(self, w, q, x):

        return self.p_G_p_q_func(w, q, x)

    def compute_grad_G_design(self, w, q, x):

        return self.p_G_p_x_func(w, q, x)

    def compute_grad_ctrl(self, w, q, x, t):
    
        return self.p_res_p_q_func(w, q, x, t)

    def compute_grad_design(self, w, q, x, t):

        return self.p_res_p_x_func(w, q, x, t)

    def get_state_size(self):

        return self.nstate
        
    def get_ctrl_size(self):

        return self.nctrl

    def reset_cache(self):
        if self.reset_cache_func is not None:
            self.reset_cache_func()
        return None


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
    J = jacobian(res_func, 0)
    G = jacobian(res_func, 1)
    p_res_p_x_func = jacobian(res_func, 2)

    pJ_pw = jacobian(J, 0)
    pJ_pq = jacobian(J, 1)
    pJ_px = jacobian(J, 2)

    pG_pw = jacobian(G, 0)
    pG_pq = jacobian(G, 1)
    pG_px = jacobian(G, 2)

    # Construct the object
    ndof = 4
    nctrl = 1
    res = residual(res_func, ndof, nctrl, p_res_p_w_func=J, p_res_p_q_func=G, p_res_p_x_func=p_res_p_x_func)

    # Test
    w0 = np.zeros(4)
    w0[0] = -1.0
    w0[2] = 0.0

    q = np.array([1.0])

    m = 2.0
    M = 7.5
    L = 1.0

    x = np.array([m, M, L])
    t = 0.
    
    res_val = res.compute(w0, q, x, t)
    d_res_d_w_val = res.compute_grad_state(w0, q, x, t)
    d_res_d_q_val = res.compute_grad_ctrl(w0, q, x, t)
    d_res_d_x_val = res.compute_grad_design(w0, q, x, t)

    print("res_val", res_val)
    print("d_res_d_w_val", d_res_d_w_val)
    print("d_res_d_q_val", d_res_d_q_val)
    print("d_res_d_x_val", d_res_d_x_val)

    pJ_pw_val = pJ_pw(w0, q, x, t)
    pJ_pq_val = pJ_pq(w0, q, x, t)
    pJ_px_val = pJ_px(w0, q, x, t)

    pG_pw_val = pG_pw(w0, q, x, t)
    pG_pq_val = pG_pq(w0, q, x, t)
    pG_px_val = pG_px(w0, q, x, t)

    print("pJ_pw_val", pJ_pw_val)
    print("pJ_pq_val", pJ_pq_val)
    print("pJ_px_val", pJ_px_val)

    print("pG_pw_val", pG_pw_val)
    print("pG_pq_val", pG_pq_val)
    print("pG_px_val", pG_px_val)

    print(pJ_pw_val.shape)
    print(pJ_pq_val.shape)
    print(pJ_px_val.shape)



    








