import numpy as np
import math
from numpy import linalg
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from utils import *

class State:
    def __init__(self,x,y,V,th):
        self.x = x
        self.y = y
        self.V = V
        self.th = th

    @property
    def xd(self):
        return self.V*np.cos(self.th)

    @property
    def yd(self):
        return self.V*np.sin(self.th)


def compute_traj_coeffs(initial_state, final_state, tf):
    """
    Inputs:
        initial_state (State)
        final_state (State)
        tf (float) final time
    Output:
        coeffs (np.array shape [8]), coefficients on the basis functions

    Hint: Use the np.linalg.solve function.
    """
    ########## Code starts here ##########
    
    t0 = 0.0
    #tf = 15.0
    v0 = initial_state.V
    vf = final_state.V
    theta_0 = initial_state.th
    theta_f = final_state.th
    xdot_0 = v0 * math.cos( theta_0 )
    ydot_0 = v0 * math.sin( theta_0 )
    xdot_f = vf * math.cos( theta_f )
    ydot_f = vf * math.sin( theta_f )
    x0 = initial_state.x
    y0 = initial_state.y
    xf = final_state.x
    yf = final_state.y
    
    psi_0 = [ 1.0, t0, t0**2, t0**3 ]
    psi_f = [ 1.0, tf, tf**2, tf**3 ]
    psidot_0 = [ 0,  1.0,  2.0 * t0,  3.0 * t0**2 ]
    psidot_f = [ 0,  1.0,  2.0 * tf,  3.0 * tf**2 ]
    
    a = np.array( [ psi_0, psi_f, psidot_0, psidot_f ] )
    bx = np.array( [ x0, xf, xdot_0, xdot_f ] )
    xcoeffs = np.linalg.solve(a, bx)
    by = np.array( [ y0, yf, ydot_0, ydot_f ] )
    ycoeffs = np.linalg.solve(a, by)
    
    #coeffs = np.vstack(( xcoeffs, ycoeffs ))
    coeffs = np.concatenate((xcoeffs, ycoeffs), axis=None)

    
    ########## Code ends here ##########
    return coeffs

def compute_traj(coeffs, tf, N):
    """
    Inputs:
        coeffs (np.array shape [8]), coefficients on the basis functions
        tf (float) final_time
        N (int) number of points
    Output:
        traj (np.array shape [N,7]), N points along the trajectory, from t=0
            to t=tf, evenly spaced in time
    """
    t = np.linspace(0,tf,N) # generate evenly spaced points from 0 to tf
    traj = np.zeros((N,7))
    
    ########## Code starts here ##########
    
    global Np, t
    
    Np = N

    for ii in range(N):
        ti = t[ii]
        psi_t = [ 1.0, ti, ti**2, ti**3 ]
        psidot_t = [ 0,  1.0,  2.0 * ti,  3.0 * ti**2 ]
        psidotdot_t = [ 0,  0.0,  2.0,  6.0 * ti ]

        xcoef = coeffs[ :4 ] 
        ycoef = coeffs[ 4:9 ]

        x_t = np.dot( psi_t, xcoef )
        y_t = np.dot( psi_t, ycoef )
        xdot_t = np.dot( psidot_t, xcoef )
        ydot_t = np.dot( psidot_t, ycoef )
        xdotdot_t = np.dot( psidotdot_t, xcoef )
        ydotdot_t = np.dot( psidotdot_t, ycoef )

        theta_t = np.arctan2( ydot_t, xdot_t )
 
        traj[ii,0] = x_t
        traj[ii,1] = y_t
        traj[ii,2] = theta_t
        traj[ii,3] = xdot_t
        traj[ii,4] = ydot_t
        traj[ii,5] = xdotdot_t
        traj[ii,6] = ydotdot_t
    
    ########## Code ends here ##########

    return t, traj

def compute_controls(traj):
    """
    Input:
        traj (np.array shape [N,7])
    Outputs:
        V (np.array shape [N]) V at each point of traj
        om (np.array shape [N]) om at each point of traj
    """
    ########## Code starts here ##########
    
    #dims = np.shape(traj)
    #N = dims[0]
    #print('N = ',N)
    
    global Np
   
    N = Np
    
    V = np.zeros((N))
    om = np.zeros((N))

    for ii in range(N):
        ti = t[ii]
        theta_t = traj[ii,2]
        xdot_t = traj[ii,3]
        ydot_t = traj[ii,4]
        xdotdot_t = traj[ii,5]
        ydotdot_t = traj[ii,6]

        V_t = math.sqrt( xdot_t**2 + ydot_t**2 )
        ct = math.cos( theta_t )
        st = math.sin( theta_t )
        #determinant_J = V_t
        invJ = np.array( [[ ct, st ], [ -st / V_t, ct / V_t ]] )
        accels = np.array( [[ xdotdot_t], [ydotdot_t]] )
        u = np.matmul( invJ, accels )
        omega_t = u[1]

        V[ii] = V_t
        om[ii] = omega_t    
    

    ########## Code ends here ##########

    return V, om

def compute_arc_length(V, t):
    """
    This function computes arc-length s as a function of t.
    Inputs:
        V: a vector of velocities of length T
        t: a vector of time of length T
    Output:
        s: the arc-length as a function of time. s[i] is the arc-length at time
            t[i]. This has length T.

    Hint: Use the function cumtrapz. This should take one line.
    """
    s = None
    ########## Code starts here ##########

    global Np
    
    N = Np
    
    s = np.zeros((N))
    s = cumtrapz( V, t )
    
    #print('size of V ',V.shape)
    #print('size of t ',t.shape)
    #print('size of s ',s.shape)
    
    s0 = np.zeros( (1) )
    #print('size of s0 ',s0.shape)
    
    #s = np.vstack( (s0, s) )
    s = np.hstack( (s0, s) )
    #print('size of augmented s ',s.shape)
    
    ########## Code ends here ##########
    return s

def rescale_V(V, om, V_max, om_max):
    """
    This function computes V_tilde, given the unconstrained solution V, and om.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained,
            differential flatness problem.
        om: vector of angular velocities of length T. Solution from the
            unconstrained, differential flatness problem.
    Output:
        V_tilde: Rescaled velocity that satisfies the control constraints.

    Hint: At each timestep V_tilde should be computed as a minimum of the
    original value V, and values required to ensure _both_ constraints are
    satisfied.
    Hint: This should only take one or two lines.
    """
    ########## Code starts here ##########

    global Np
    
    N = Np
    
    V_tilde = np.zeros((N))
    
    for ii in range(N):
#        if om[ii] > 1.0E-8:
        if abs( om[ii] ) > 1.0E-8:
            V_om_constr = V[ii] * np.sign( om[ii] ) * om_max / om[ii]
        else:
            V_om_constr = V[ii]   
        V_tilde[ii] = min( V[ii], V_max, V_om_constr )            
    
    ########## Code ends here ##########
    
    return V_tilde


def compute_tau(V_tilde, s):
    """
    This function computes the new time history tau as a function of s.
    Inputs:
        V_tilde: a sequence of scaled velocities of length T.
        s: a sequence of arc-length of length T.
    Output:
        tau: the new time history for the sequence. tau[i] is the time at s[i]. This has length T.

    Hint: Use the function cumtrapz. This should take one line.
    """
    ########## Code starts here ##########
    
    global Np
    
    N = Np
    
    tau = np.zeros((N))
    
    tau = cumtrapz( np.reciprocal( V_tilde ), s )
    
    #print('tau[0] ',tau[0])
    tau0 = np.zeros( (1) )
    
    #tau = np.vstack( (s0, s) )
    tau = np.hstack( (tau0, tau) )

    #print('size of tau ',tau.shape)
    #print('tau[0] ',tau[0])

    ########## Code ends here ##########
    
    return tau

def rescale_om(V, om, V_tilde):
    """
    This function computes the rescaled om control.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained, differential flatness problem.
        om:  vector of angular velocities of length T. Solution from the unconstrained, differential flatness problem.
        V_tilde: vector of scaled velocities of length T.
    Output:
        om_tilde: vector of scaled angular velocities

    Hint: This should take one line.
    """
    ########## Code starts here ##########
    
    global Np
    
    N = Np
    
    om_tilde = np.zeros((N))
    
    for ii in range(N):
        om_tilde[ii] = om[ii] * V_tilde[ii] / V[ii]            
    

    ########## Code ends here ##########
    
    return om_tilde

def compute_traj_with_limits(z_0, z_f, tf, N, V_max, om_max):
    coeffs = compute_traj_coeffs(initial_state=z_0, final_state=z_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)

    return traj, tau, V_tilde, om_tilde

def interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f):
    """
    Inputs:
        traj (np.array [N,7]) original unscaled trajectory
        tau (np.array [N]) rescaled time at orignal traj points
        V_tilde (np.array [N]) new velocities to use
        om_tilde (np.array [N]) new rotational velocities to use
        dt (float) timestep for interpolation

    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    """
    # Get new final time
    tf_new = tau[-1]

    # Generate new uniform time grid
    N_new = int(tf_new/dt)
    t_new = dt*np.array(range(N_new+1))

    
    #print('size of t_new ',t_new.shape)
    #print('size of tau ',tau.shape)
    #print('size of traj ',traj.shape)

    # Interpolate for state trajectory
    traj_scaled = np.zeros((N_new+1,7))
    traj_scaled[:,0] = np.interp(t_new,tau,traj[:,0])   # x
    traj_scaled[:,1] = np.interp(t_new,tau,traj[:,1])   # y
    traj_scaled[:,2] = np.interp(t_new,tau,traj[:,2])   # th
    # Interpolate for scaled velocities
    V_scaled = np.interp(t_new, tau, V_tilde)           # V
    om_scaled = np.interp(t_new, tau, om_tilde)         # om
    # Compute xy velocities
    traj_scaled[:,3] = V_scaled*np.cos(traj_scaled[:,2])    # xd
    traj_scaled[:,4] = V_scaled*np.sin(traj_scaled[:,2])    # yd
    # Compute xy acclerations
    traj_scaled[:,5] = np.append(np.diff(traj_scaled[:,3])/dt,-s_f.V*om_scaled[-1]*np.sin(s_f.th)) # xdd
    traj_scaled[:,6] = np.append(np.diff(traj_scaled[:,4])/dt, s_f.V*om_scaled[-1]*np.cos(s_f.th)) # ydd

    return t_new, V_scaled, om_scaled, traj_scaled

if __name__ == "__main__":
    # traj, V, om = differential_flatness_trajectory()
    # Constants
    tf = 15.
    V_max = 0.5
    om_max = 1

    # time
    dt = 0.005
    N = int(tf/dt)+1
    t = dt*np.array(range(N))

    # Initial conditions
    s_0 = State(x=0, y=0, V=V_max, th=-np.pi/2)

    # Final conditions
    s_f = State(x=5, y=5, V=V_max, th=-np.pi/2)

    coeffs = compute_traj_coeffs(initial_state=s_0, final_state=s_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj)

    part_b_complete = False
    s = compute_arc_length(V, t)
    if s is not None:
        part_b_complete = True
        V_tilde = rescale_V(V, om, V_max, om_max)
        tau = compute_tau(V_tilde, s)
        om_tilde = rescale_om(V, om, V_tilde)

        t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)

        # Save trajectory data
        data = {'z': traj_scaled, 'V': V_scaled, 'om': om_scaled}
        save_dict(data, "data/differential_flatness.pkl")

    maybe_makedirs('plots')

    # Plots
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    plt.plot(traj[:,0], traj[:,1], 'k-',linewidth=2)
    plt.grid(True)
    plt.plot(s_0.x, s_0.y, 'go', markerfacecolor='green', markersize=15)
    plt.plot(s_f.x, s_f.y, 'ro', markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title("Path (position)")
    plt.axis([-1, 6, -1, 6])

    ax = plt.subplot(2, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
    plt.title('Original Control Input')
    plt.tight_layout()

    plt.subplot(2, 2, 4, sharex=ax)
    if part_b_complete:
        plt.plot(t_new, V_scaled, linewidth=2)
        plt.plot(t_new, om_scaled, linewidth=2)
        plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
        plt.grid(True)
    else:
        plt.text(0.5,0.5,"[Problem iv not completed]", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.xlabel('Time [s]')
    plt.title('Scaled Control Input')
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    if part_b_complete:
        h, = plt.plot(t, s, 'b-', linewidth=2)
        handles = [h]
        labels = ["Original"]
        h, = plt.plot(tau, s, 'r-', linewidth=2)
        handles.append(h)
        labels.append("Scaled")
        plt.legend(handles, labels, loc="best")
    else:
        plt.text(0.5,0.5,"[Problem iv not completed]", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Arc-length [m]')
    plt.title('Original and scaled arc-length')
    plt.tight_layout()
    plt.savefig("plots/differential_flatness.png")
    plt.show()
