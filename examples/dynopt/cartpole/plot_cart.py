import matplotlib
import matplotlib.pyplot as plt
import niceplots
import numpy as np

# TODO:
# 1. Add design variable interface
# 2. Color
# 3. Plot force.
# 4. Fix sign

my_blue = "#4C72B0"
my_red = "#C54E52"
my_green = "#56A968"
my_brown = "#b4943e"
my_purple = "#684c6b"
my_orange = "#cc5500"

class plot_invert_pendulum(object):

    def __init__(self, x_design_dict, ax):

        self.ax = ax

        self.d = 2.0
        self.h0 = 0.2
        self.h = x_design_dict["M"] / 5.0

        self.radius = self.h0

        self.l_bar = x_design_dict["L"]
        self.ball_radius = x_design_dict["m"] / 1.0 * 0.3
        

    def set_u(self, u):

        l_bar = self.l_bar
        h0 = self.h0
        h = self.h

        self.x = u[0]
        self.alpha = u[1]


    def plot_one_instance(self, x_ball_arr, y_ball_arr, force):

        self.plot_ground()
        self.plot_cart()
        self.plot_wheel()
        self.plot_bar()
        self.plot_trajectory(x_ball_arr, y_ball_arr)
        self.plot_force(force)

        return self.ax

    def plot_trajectory(self, x_ball_arr, y_ball_arr):

        ax = self.ax

        ax.plot(x_ball_arr, y_ball_arr, 'k', alpha = 0.3)

    def plot_ground(self):

        ax = self.ax

        ax.plot([-12, 12], [0, 0], 'k')

    def plot_cart(self):

        x = self.x

        d = self.d
        h0 = self.h0
        h = self.h

        ax = self.ax

        x_edge = [x - d / 2, x + d / 2, x + d / 2, x - d / 2]
        y_edge = [h0, h0, h0 + h, h0 + h]

        ax.fill(x_edge, y_edge, color = my_blue) 

    def plot_force(self, force):

        ax = self.ax

        force_scale = 10

        x = self.x

        d = self.d
        h0 = self.h0
        h = self.h

        ax.arrow(x, h0 + h / 2, force / force_scale, 0, color = my_red, head_width = 0.3, head_length = 0.5)

    def plot_wheel(self):

        ax = self.ax 

        x = self.x

        d = self.d
        h0 = self.h0
        h = self.h

        radius = self.radius

        alpha = x / radius

        # Left
        circle1 = plt.Circle((x - d / 2 * 0.6, h0), radius=radius, color='k')
        ax.add_patch(circle1)
        circle1 = plt.Circle((x - d / 2 * 0.6, h0), radius=radius*0.9, color='w')
        ax.add_patch(circle1)
        ax.plot([x - d / 2 * 0.6, x - d / 2 * 0.6 + np.sin(alpha) * radius], [h0, h0 + np.cos(alpha) * radius], 'k')

        # Right
        circle2 = plt.Circle((x + d / 2 * 0.6, h0), radius=radius, color='k')
        ax.add_patch(circle2)
        circle2 = plt.Circle((x + d / 2 * 0.6, h0), radius=radius*0.9, color='w')
        ax.add_patch(circle2)
        ax.plot([x + d / 2 * 0.6, x + d / 2 * 0.6 + np.sin(alpha) * radius], [h0, h0 + np.cos(alpha) * radius], 'k')

    def plot_bar(self):

        x = self.x

        alpha = self.alpha

        ax = self.ax 

        d = self.d
        h0 = self.h0
        h = self.h

        ball_radius = self.ball_radius

        l_bar = self.l_bar

        

        ax.plot([x, x+ (l_bar - ball_radius) * np.sin(alpha)], [h0 + h / 2, h0 + h / 2 - (l_bar - ball_radius) * np.cos(alpha)], 'k')
        circle = plt.Circle((x+ l_bar * np.sin(alpha), h0 + h / 2 - l_bar * np.cos(alpha)), radius=ball_radius, color='r')
        ax.add_patch(circle)


if __name__ == "__main__":

    fig, ax = plt.subplots()

    niceplots.setRCParams()
    niceplots.All()

    ax.set_aspect(1)

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 5])

    invert_pendulum_instance = plot_invert_pendulum(ax)

    u = np.array([1.1, np.pi/3])
    invert_pendulum_instance.set_u(u)
    ax = invert_pendulum_instance.plot_one_instance()

    plt.show()