"""
==================================
Inverted pendulum animation class
==================================

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from random import random, seed
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation

class EnvAnimate:
    '''
    Initialize Inverted Pendulum Animation Settings
    '''
    def __init__(self):       
        pass
        
    def load_random_test_trajectory(self,):
        # Random trajectory for example
        self.theta = np.linspace(-np.pi, np.pi, self.t.shape[0])
        self.u = np.zeros(self.t.shape[0])

        self.x1 = np.sin(self.theta)
        self.y1 = np.cos(self.theta)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])

        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.8, '', transform=self.ax.transAxes)
        pass
    
    '''
    Provide new rollout trajectory (theta and control input) to reanimate
    '''
    def load_trajectory(self, theta, u):
        """
        Once a trajectory is loaded, you can run start() to see the animation
        ----------
        theta : 1D numpy.ndarray
            The angular position of your pendulum (rad) at each time step
        u : 1D numpy.ndarray
            The control input at each time step
            
        Returns
        -------
        None
        """
        self.theta = theta
        self.x1 = np.sin(theta)
        self.y1 = np.cos(theta)
        self.u = u
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2,-2, 2])
        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
        pass
    
    # region: Animation
    # Feel free to edit (not necessarily)
    def init(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text

    def _update(self, i):
        thisx = [0, self.x1[i]]
        thisy = [0, self.y1[i]] 
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (self.t[i], self.theta[i], self.u[i]))
        return self.line, self.time_text

    def start(self, filename):
        print('Starting Animation')
        print()
        # Set up plot to call animate() function periodically
        self.ani = FuncAnimation(self.fig, self._update, frames=range(len(self.x1)), interval=100, blit=True, init_func=self.init, repeat=False)
        self.ani.save(filename)
        plt.show()
    # endregion: Animation


class InvertedPendulum(EnvAnimate):
    def __init__(self):
        EnvAnimate.__init__(self,)
        # Change this to match your discretization
        # Usually, you will need to load parameters including
        # constants: dt, vmax, umax, n1, n2, nu
        # parameters: a, b, σ, k, r, γ
        
        self.dt = 0.1 # time step
        self.t = np.arange(0.0, 5.0, self.dt)
        self.load_random_test_trajectory()
        self.gamma = 0.5
        
        self.v_max = 4
        self.u_max = 3
        self.n1 = 73 # 5 degree
        self.n2 = 41 
        self.nu = 31
        self.u_list = np.linspace(-self.u_max, self.u_max, self.nu)
        
        self.k = 3 # shape of cost
        self.r = 0.01 # scale control cost
        self.a = 1 # effect of gravity, mass adn length of pendulum
        self.b = 0.5 # damping and friction
        # sigma_1 = 0.4, sigma_2 = 2
        self.sigma = np.array([0.05, 0.1]).reshape((2,1))
                                               
    # TODO: Implement your own VI and PI algorithms
    # Feel free to edit/delete these functions
    def l_xu(self, x1, x2, u):
        '''
        Stage cost
        input index
        return stage cost
        '''
        x1, x2 = self.index2state(x1, x2)
        u = self.u_list[u]
        return 1 - np.exp(self.k * np.cos(x1) - self.k) + self.r / 2 * (u**2)
    
    def f_xu(self, x1, x2, u, animate = False):
        '''
        Motion model
        input index
        return probability for each state in index
        '''
        x1, x2 = self.index2state(x1, x2)
        if animate == False: u = self.u_list[u]
        
        mean = np.zeros(2)
        mean[0] = x1 + x2 * self.dt
        if mean[0] > np.pi: mean[0] = np.pi
        if mean[0] < -np.pi: mean[0] = -np.pi
        mean[1] = x2 + (self.a * np.sin(x1) - self.b * x2 + u) * self.dt
        if mean[1] > self.v_max: mean[1] = self.v_max
        if mean[1] < -self.v_max: mean[1] = -self.v_max
        cov = self.sigma * self.sigma.T
        
        x,y = np.mgrid[-np.pi:np.pi+2*np.pi/(self.n1-1):2*np.pi/(self.n1-1),
                       -self.v_max:self.v_max+2*self.v_max /(self.n2-1): 2*self.v_max /(self.n2-1)]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        rv = multivariate_normal(mean = mean, cov = cov+0.001)
        
        return rv.pdf(pos) / np.sum(rv.pdf(pos))
    
    def index2state(self, x1, x2):
        '''
        input index
        return true state value
        '''
        theta = x1 * (2*np.pi / (self.n1-1)) - np.pi
        angular_v = x2 * 2*self.v_max /(self.n2-1) - self.v_max
        return theta, angular_v
    
    def value_iteration(self,):
        print('Running value iteration...')
        # angle, angular velocity and control
        V_x = np.ones((self.n1, self.n2)) * 0
        VI_policy = np.zeros((self.n1, self.n2))
        V_x[ int((self.n1-1)/2), int((self.n2-1)/2)] = 0
        V_x_t = [V_x]
        
        count = 0
        while(True):
            start = time()
            count += 1
            print('Iteration ', count ,'...')
            V_x_1 = np.zeros((self.n1, self.n2))
            for i in range(V_x.shape[0]):
                for j in range(V_x.shape[1]):
                    curr = np.zeros((len(self.u_list)))
                    # calculate new value function
                    for k in range(len(self.u_list)):
                        cost = self.l_xu(i,j,k)
                        x_prime = self.f_xu(i,j,k)
                        # calculate sum of value function
                        V_sum = np.sum(np.multiply(V_x, x_prime))
                        curr[k] = cost + self.gamma * V_sum
                    u_min = np.argmin(curr)
                    V_x_1[i,j] = curr[u_min]
                    VI_policy[i,j] = u_min
            end = time()
            print('Iteration time:', (end - start))
            # terminal threshold
            if np.all(abs(V_x_1 - V_x) < 0.01):
                break            
            V_x = V_x_1
            V_x_t.append(V_x)
        return V_x_t, VI_policy
    
    def policy_iteration(self,):
        print('Running policy iteration...')
        V_x = np.zeros((self.n1, self.n2))
        PI_policy = np.zeros((self.n1, self.n2))
        V_x_t = [V_x]
        
        count = 0
        while(True):
            start = time()
            count += 1
            print('Iteration ', count ,'...')
            
            # Policy Evaluation
            V_x_pi = np.zeros((self.n1, self.n2))
            for i in range(V_x.shape[0]):
                for j in range(V_x.shape[1]):
                    control = int(PI_policy[i,j])
                    x_prime = self.f_xu(i, j, control)
                    cost = self.l_xu(i, j, control)
                    V_sum = np.sum(np.multiply(x_prime, V_x))
                    V_x_pi[i,j] = cost + self.gamma * V_sum
                        
            # Policy Improvement
            for i in range(V_x.shape[0]):
                for j in range(V_x.shape[1]):
                    curr = np.zeros((len(self.u_list)))
                    # calculate new value function
                    for k in range(len(self.u_list)):
                        cost = self.l_xu(i, j, k)
                        x_prime = self.f_xu(i, j, k)
                        V_sum = np.sum(np.multiply(x_prime, V_x_pi))
                        curr[k] = cost + self.gamma * V_sum
                    u_min = np.argmin(curr)
                    V_x[i,j] = curr[u_min]
                    PI_policy[i,j] = u_min
            
            end = time()
            print('Iteration time:', (end - start))
            # terminal threshold
            if np.all(abs(V_x - V_x_pi) < 0.01):
                break            
            V_x_pi = V_x
            V_x_t.append(V_x)
        return V_x_t, PI_policy
    
    def generate_trajectory(self, policy):
        theta = np.zeros(self.t.shape[0])
        u = np.zeros(self.t.shape[0])
        
        # generate initial state randomly
        seed(time())
        x1 = random() * 2 * np.pi - np.pi
        x2 = random() * 2 * self.v_max - self.v_max
        
        for t in range(len(self.t)):
            if t == 0: x1, x2 = self.interpolate(x1, x2)
            control = int(policy[x1, x2])
            theta[t], _= self.index2state(x1,x2)
            u[t] = self.u_list[control]
            x_prime = self.f_xu(x1, x2, control)
            next_x1, next_x2 = None, None
            next_val = 0
            for i in range(x_prime.shape[0]):
                for j in range(x_prime.shape[1]):
                    curr = x_prime[i,j] * random() 
                    if curr > next_val :
                        next_val = curr
                        next_x1 = i
                        next_x2 = j
            x1, x2 = next_x1, next_x2
            
        return theta, u
    
    def interpolate(self, x1, x2):
        x1 = round((x1 + np.pi) / (2 * np.pi) * self.n1-1)
        x2 = round((x2 + self.v_max) / (2 * self.v_max) * self.n2-1)
        return x1, x2


if __name__ == '__main__':
    inv_pendulum = InvertedPendulum()
    
    # Solve for policy
    VI_v, VI_policy = inv_pendulum.value_iteration()
    PI_v, PI_policy = inv_pendulum.policy_iteration()
    
    for i in range(10):
        # generate VI policy trajectory
        vi_theta, vi_u = inv_pendulum.generate_trajectory(VI_policy)
        inv_pendulum.load_trajectory(vi_theta, vi_u)
        inv_pendulum.start('VI_' + str(i+1) + '.gif')
        
        # generate PI policy trajectory
        pi_theta, pi_u = inv_pendulum.generate_trajectory(PI_policy)
        inv_pendulum.load_trajectory(pi_theta, pi_u)
        inv_pendulum.start('PI_' + str(i+1) + '.gif')
    
    
    ######## TODO: Implement functions for visualization ########
    #############################################################