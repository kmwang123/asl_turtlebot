import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    
    tck, u = scipy.interpolate.splprep(np.array(path).transpose(), s=alpha)
    u = np.linspace(0, 1.00, 1000)
    x, y = scipy.interpolate.splev(u, tck)
    path_len = np.sum(np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2))
    t_move = path_len / V_des
    t_smoothed = np.append(np.arange(0, t_move, dt), t_move)
    u_new = t_smoothed/t_move
    x, y = scipy.interpolate.splev(u_new, tck)
    xd, yd = scipy.interpolate.splev(u_new, tck, der=1) / t_move
    xdd, ydd = scipy.interpolate.splev(u_new, tck, der=2) / t_move**2
    th = np.arctan2(yd,xd)
#    xd = yd = xdd = ydd = th = np.zeros(len(x))

    traj_smoothed = np.array([x, y, th, xd, yd, xdd, ydd]).transpose()
    """
    N = len( path )
    #print('N = ',N)

    # generate a time vector for the given points assuming constant velocity V_des

    tvec = np.zeros( (N) )

    for itr1 in range( 1, N ) :
        
        x = path[itr1]
        xlast = path[itr1-1]
        #print(x)
        #print(x[0],x[1])
        
        delta_dist = np.sqrt( ( x[0] - xlast[0] )**2 + ( x[1] - xlast[1]  )**2  ) 
        delta_t = delta_dist / V_des
        tvec[ itr1 ] = tvec[ itr1 - 1 ] + delta_t

    # get the splines for x and y data vs time

    path_array = np.asarray( path )
     
    splinex = scipy.interpolate.splrep( tvec, path_array[:,0] )
    spliney = scipy.interpolate.splrep( tvec, path_array[:,1] )

    # create an evenly spaced time vector to evaluate the splines

    tmax = tvec[ N - 1 ]
    Numpts = int( tmax / dt ) + 1

    #tvec_even = np.arange( 0, tmax, tstep )
    tvec_even = np.linspace( 0, tmax, Numpts )

    # evaluate the splines for x and y 
    
    x = scipy.interpolate.splev( tvec_even, splinex )
    y = scipy.interpolate.splev( tvec_even, spliney )
 
    # evaluate the splines for first derivatives

    xdot = scipy.interpolate.splev( tvec_even, splinex, 1 )
    ydot = scipy.interpolate.splev( tvec_even, spliney, 1 )

    # evaluate the splines for second derivatives

    xddot = scipy.interpolate.splev( tvec_even, splinex, 2 )
    yddot = scipy.interpolate.splev( tvec_even, spliney, 2 )

    # compute theta trajectory from atan2 of ydot / xdot

    theta = np.zeros( Numpts )

    for itr1 in range( Numpts ) :
        theta[ itr1 ] = np.arctan2( ydot[ itr1], xdot[ itr1 ] )

    # pack the data into the return variables

    traj_smoothed = np.zeros( ( Numpts, 7 ) )
    traj_smoothed[:,0] = x
    traj_smoothed[:,1] = y
    traj_smoothed[:,2] = theta
    traj_smoothed[:,3] = xdot
    traj_smoothed[:,4] = ydot
    traj_smoothed[:,5] = xddot
    traj_smoothed[:,6] = yddot

    t_smoothed = tvec_even #np.array(N)
    """

    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
