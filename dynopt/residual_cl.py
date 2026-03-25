import autograd.numpy as np
from autograd import jacobian, grad, elementwise_grad
from . import residual

"""
    Closed loop residual.
    For dynamical system of the form
        wgt_dot + deltaw_dot = r(wtgt + deltaw, q_tgt + deltaq, x) 
    =>  deltaw_dot = r(wtgt + deltaw, q_tgt + deltaq, x) 
    where deltaq = W deltaw.

    # Inputs: 
    * residual: Residual object.
    * W: Linking perturbed control var, deltaq to perturbed state deltaw.
    * wtgt: Steady-state target state var.
    * qtgt: Steady-state target control var.
    * x: design variable.

"""

class residual_cl:

    def __init__(self, res_obj: residual.residual, W, wtgt, qtgt):

        self.res_obj = res_obj

        self.W = W

        self.wtgt = wtgt
        self.qtgt = qtgt

    def compute(self, deltaw, x, t):

        res_obj = self.res_obj

        wtgt = self.wtgt
        qtgt = self.qtgt

        W = self.W

        w = wtgt + deltaw
        q = qtgt + W.dot(deltaw)

        r = res_obj.compute(w, q, x, t)

        return r

    def compute_grad_state(self, deltaw, x, t):

        res_obj = self.res_obj

        wtgt = self.wtgt
        qtgt = self.qtgt

        W = self.W

        w = wtgt + deltaw
        q = qtgt + W.dot(deltaw)

        p_r_p_w = res_obj.compute_grad_state(w, q, x, t)
        p_r_p_q = res_obj.compute_grad_ctrl(w, q, x, t)
        
        p_r_p_deltaw = p_r_p_w + p_r_p_q.dot(W)
        return p_r_p_deltaw

    def compute_grad_design(self, deltaw, x, t):

        res_obj = self.res_obj

        wtgt = self.wtgt
        qtgt = self.qtgt

        W = self.W

        w = wtgt + deltaw
        q = qtgt + W.dot(deltaw)

        p_r_p_x = res_obj.compute_grad_design(w, q, x, t)

        return p_r_p_x

    def compute_grad_inputs_b(self, deltaw, x, res_b, t):

        res_obj = self.res_obj

        wtgt = self.wtgt
        qtgt = self.qtgt

        W = self.W

        w = wtgt + deltaw
        q = qtgt + W.dot(deltaw)

        w_b = res_obj.compute_grad_state(w, q, x, t).T.dot(res_b)
        q_b = res_obj.compute_grad_ctrl(w, q, x, t).T.dot(res_b)

        wtgt_b = w_b
        qtgt_b = q_b
        W_b = np.outer(q_b, deltaw)

        return W_b, wtgt_b, qtgt_b

    def get_state_size(self):

        return self.res_obj.get_state_size()
    
    def get_ctrl_size(self):

        return self.res_obj.get_ctrl_size()

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

    np.random.seed(10)

    W = np.random.rand(1, 4)
    wtgt = np.array([0.0, np.pi, 0.0, 0.0])
    qtgt = np.array([0.0])

    res_cl = residual_cl(res, W, wtgt, qtgt)

    # Test
    deltaw = np.random.rand(4)

    m = 2.0
    M = 7.5
    L = 1.0

    x = np.array([m, M, L])
    t = 0.

    res_val = res_cl.compute(deltaw, x, t)
    d_res_d_deltaw_val = res_cl.compute_grad_state(deltaw, x, t)
    d_res_d_x_val = res_cl.compute_grad_design(deltaw, x, t)

    print("res_val", res_val)
    print("d_res_d_deltaw_val", d_res_d_deltaw_val)
    print("d_res_d_x_val", d_res_d_x_val)