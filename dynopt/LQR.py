# LQR Linear quadratic regulator and adjoint
import scipy.linalg
import autograd.numpy as np
from autograd import jacobian, grad, elementwise_grad
import copy


class LQR(object):
    def __init__(self, A, B, Q, R):

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def solve(self):

        A = self.A
        B = self.B
        Q = self.Q
        R = self.R

        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R).dot(B.T.dot(P)) # (-W)

        self.P = P
        self.K = K

        return self.P

    def solve_adjoint(self, pfpP_value):
        
        A = self.A
        B = self.B
        K = self.K

        QQ = -0.5 * (pfpP_value + pfpP_value.T)
        Atilt = A - B.dot(K)

        Psi = scipy.linalg.solve_continuous_lyapunov(Atilt, QQ)

        self.Psi = Psi

        return Psi

    def compute_grad_input_b(self, R_b):

        A = self.A
        B = self.B
        Q = self.Q
        R = self.R

        P = self.P

        A_b = P.dot(R_b.T) + P.T.dot(R_b)
        B_b = - ((P.T.dot(R_b)).dot(P.T).dot(B)).dot(np.linalg.inv(R).T) - P.dot(R_b).dot(P).dot(B).dot(np.linalg.inv(R))

        return A_b, B_b

if __name__ == "__main__":

    # HACK: Only works with x = A, B (labeled AAA BBB)
    class FoI_LQR(object):

        def __init__(self, FoI_func, p_FoI_p_P_func=None, p_FoI_p_x_func=None):

            self.FoI_func = FoI_func

            self.p_FoI_p_P_func = p_FoI_p_P_func
            self.p_FoI_p_x_func = p_FoI_p_x_func
            
        def compute(self, P, x):

            return self.FoI_func(P, x)

        def compute_grad_state(self, P, x):

            return self.p_FoI_p_P_func(P, x)

        def compute_grad_design(self, P, x):

            return self.p_FoI_p_x_func(P, x)    

    class implicit_FoI_LQR(object):
        
        def __init__(self, FoI_lqr: FoI_LQR, lqr: LQR, x):

            self.FoI_lqr = FoI_lqr
            self.lqr = lqr

            self.x = x

            # AAA
            self.lqr.A = x

            # BBB
            # self.lqr.B = x

        def set_design(self, x):

            self.x = x

            # AAA
            self.lqr.A = x

            # BBB
            # self.lqr.B = x

        def solve(self):

            self.P = self.lqr.solve()

            return self.P

        def solve_adjoint(self):

            FoI_lqr = self.FoI_lqr
            lqr = self.lqr
            x = self.x

            P = self.P

            p_FoI_p_P = FoI_lqr.compute_grad_state(P, x)
            print("p_FoI_p_P", p_FoI_p_P)

            self.Psi = lqr.solve_adjoint(p_FoI_p_P)

        def compute(self):

            FoI_lqr = self.FoI_lqr
            lqr = self.lqr
            x = self.x

            P = self.P

            FoI_lqr_val = FoI_lqr.compute(P, x)

            return FoI_lqr_val

        def compute_grad_x(self):

            FoI_lqr = self.FoI_lqr
            lqr = self.lqr
            x = self.x

            P = self.P
            Psi = self.Psi

            p_FoI_p_x_1 = FoI_lqr.compute_grad_design(P, x)
            # AAA
            p_FoI_p_x_2 = lqr.compute_grad_input_b(Psi)[0]
            # BBB
            #p_FoI_p_x_2 = lqr.compute_grad_input_b(Psi)[1]

            print("p_FoI_p_x_1", p_FoI_p_x_1)

            d_FoI_d_x = p_FoI_p_x_1 + p_FoI_p_x_2

            return d_FoI_d_x



    # ===================
    # Analyis
    # ===================
    l = 2.0  # rod length is 2l
    m = 1.0 #(2 * l) * (0.006 ** 2) * (3.14 / 4) * 7856  # rod 6 mm diameter, 44cm length, 7856 kg/m^3
    M = 5.0
    dt = 0.02  # 20 ms
    g = 10

    A = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, g / (l * (4 / 3 - m / (m + M))), 0],
            [0, 0, 0, 1],
            [0, 0, g / (l * (4 / 3 - m / (m + M))), 0],
        ]
    )
    B = np.array([[0], [1 / (m + M)], [0], [-1 / (l * (4 / 3 - m / (m + M)))]])
    Q = np.array([[1, 0, 0, 0], [0, 0.0001, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0.0001]])
    #R = np.array([[0.0005]])
    R = np.eye(1) * 0.0005
    #print('B', B.shape)
    #print('R', R.shape)

    # Prime solution
    lqr = LQR(A, B, Q, R)
    P = lqr.solve()
    # lqr_obj.set_f(f)
    # lqr_obj.compute_f()
    # f_value = lqr_obj.get_f()

    # ===================
    # Adjoint
    # ===================

    # Compute derivative

    def FoI_func(P, x):
        return np.sum(P) + np.sum(x)

    p_FoI_p_P_func = jacobian(FoI_func, 0)
    p_FoI_p_x_func = jacobian(FoI_func, 1)

    FoI_lqr = FoI_LQR(FoI_func, p_FoI_p_P_func = p_FoI_p_P_func, p_FoI_p_x_func = p_FoI_p_x_func)

    # AAA
    imp_FoI_lqr = implicit_FoI_LQR(FoI_lqr, lqr, A)

    # BBB
    #imp_FoI_lqr = implicit_FoI_LQR(FoI_lqr, lqr, B)


    imp_FoI_lqr.solve()
    FoI = imp_FoI_lqr.compute()
    imp_FoI_lqr.solve_adjoint()
    # AAA
    d_FoI_d_A = imp_FoI_lqr.compute_grad_x()
    # BBB
    #d_FoI_d_B = imp_FoI_lqr.compute_grad_x()

    epsilon = 1e-6
    # AAA
    d_FoI_d_A_FD = np.zeros_like(d_FoI_d_A)
    # BBB
    #d_FoI_d_B_FD = np.zeros_like(d_FoI_d_B)
    for i in range(d_FoI_d_A.shape[0]):
        for j in range(d_FoI_d_A.shape[1]):
    #for i in range(d_FoI_d_B.shape[0]):
        #for j in range(d_FoI_d_B.shape[1]):

            # AAA
            A_p = copy.deepcopy(A)
            A_p[i, j] += epsilon

            imp_FoI_lqr.set_design(A_p)

            # BBB
            #B_p = copy.deepcopy(B)
            #B_p[i, j] += epsilon

            #imp_FoI_lqr.set_design(B_p)

            
            imp_FoI_lqr.solve()

            FoI_p = imp_FoI_lqr.compute()

            # AAA
            d_FoI_d_A_FD[i, j] = (FoI_p - FoI) / epsilon

            # BBB
            #d_FoI_d_B_FD[i, j] = (FoI_p - FoI) / epsilon

    # AAA
    print("d_FoI_d_A", d_FoI_d_A)      
    print("d_FoI_d_A_FD", d_FoI_d_A_FD)      

    # BBB
    #print("d_FoI_d_B", d_FoI_d_B)      
    #print("d_FoI_d_B_FD", d_FoI_d_B_FD)      







    # lqr_obj.set_pfpP(pfpP)
    # lqr_obj.compute_pfpP()
    # lqr_obj.solve_adjoint()
    # lqr_obj.compute_dfdx()

    # # Extract derivative
    # [P, K] = lqr_obj.get_sol()
    # [dfdA, dfdB, dfdQ, dfdR] = lqr_obj.get_dfdx()

    # print("dfdA", dfdA)
    # print("dfdB", dfdB)
    # print("dfdQ", dfdQ)
    # print("dfdR", dfdR)

    # # ===================
    # # Finite difference
    # # ===================
    # # dfdA FD
    # epsilon = 1e-6
    # A_p = copy.deepcopy(A)
    # A_p[0, 0] += epsilon
    # lqr_obj = lqr(A_p, B, Q, R)

    # lqr_obj.solve()
    # lqr_obj.set_f(f)
    # lqr_obj.compute_f()
    # f_value_p = lqr_obj.get_f()

    # dfdA_FD = (f_value_p - f_value) / epsilon

    # # dfdB FD
    # epsilon = 1e-6
    # B_p = copy.deepcopy(B)
    # B_p[0, 0] += epsilon
    # lqr_obj = lqr(A, B_p, Q, R)

    # lqr_obj.solve()
    # lqr_obj.set_f(f)
    # lqr_obj.compute_f()
    # f_value_p = lqr_obj.get_f()

    # dfdB_FD = (f_value_p - f_value) / epsilon

    # # dfdQ FD
    # epsilon = 1e-6
    # Q_p = copy.deepcopy(Q)
    # Q_p[0, 0] += epsilon
    # lqr_obj = lqr(A, B, Q_p, R)

    # lqr_obj.solve()
    # lqr_obj.set_f(f)
    # lqr_obj.compute_f()
    # f_value_p = lqr_obj.get_f()

    # dfdQ_FD = (f_value_p - f_value) / epsilon

    # # dfdR FD
    # epsilon = 1e-6
    # R_p = copy.deepcopy(R)
    # R_p[0, 0] += epsilon
    # lqr_obj = lqr(A, B, Q, R_p)

    # lqr_obj.solve()
    # lqr_obj.set_f(f)
    # lqr_obj.compute_f()
    # f_value_p = lqr_obj.get_f()

    # dfdR_FD = (f_value_p - f_value) / epsilon

    # print("      Adjoint            FD")
    # print("-" * 50)
    # print("dfdA:", dfdA[0, 0], dfdA_FD)
    # print("dfdB:", dfdB[0, 0], dfdB_FD)
    # print("dfdQ:", dfdQ[0, 0], dfdQ_FD)
    # print("dfdR:", dfdR[0, 0], dfdR_FD)
    # print("-" * 50)
