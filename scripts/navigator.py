#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from asl_turtlebot.msg import DetectedObject, DetectedObjectList
from std_msgs.msg import String
import tf
import numpy as np
from numpy import linalg
from scipy import ndimage
from utils import wrapToPi
from planners import AStar, compute_smoothed_traj
from grids import StochOccupancyGrid2D
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum
from visualization_msgs.msg import Marker

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

import waypoints
import pdb

NUM_WAYPOINTS = 9

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    STOP = 4
    CROSS = 5

class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """
    def __init__(self):
        rospy.init_node('turtlebot_navigator', anonymous=True)
        self.mode = Mode.IDLE

        #delivery lists
        self.delivery_req_list = []
        self.detected_objects_names = []
        self.detected_objects = []
        self.marker_dict = {}
        self.objectname_markerLoc_dict = {}
        self.delivery_flag = False
        self.delivery_done = False
        self.NUM_DELIVERY_ITEMS = 0
        self.home_flag = False
       

        #stop sign params
        self.stop_min_dist = 0.5
        self.stop_time = 3.
        self.crossing_time = 1.5
        self.stop_flag = 0

        #force move params
        self.move_time = 3.
        self.wait_time  =  6.0
        self.waypoint_flag = True
        self.vendor_ind = 0
        self.intermediate_goal_flag = 0

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        # intermediate goal state
        self.x_mid = None
        self.y_mid = None

        # initial state
        self.x_init = rospy.get_param("~x_pos",3.15)
        self.y_init = rospy.get_param("~y_pos",1.6)
        self.z_init = rospy.get_param("~z_pos",0.0)
        self.th_init = 0.0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0,0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False
        self.map_threshold = 40 

        # plan parameters
        self.plan_resolution =  0.04
        self.plan_horizon = 4.0
        self.state_min = self.snap_to_grid((-0.05,-0.05))
        self.state_max = self.snap_to_grid((self.plan_horizon,self.plan_horizon))

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.,0.]
        
        # Robot limits
        self.v_max = 0.21    # maximum velocity (orig is 0.2)
        self.om_max = 0.35   # maximum angular velocity (orig is 0.4)

        self.v_des = 0.06#0.12   # desired cruising velocity
        self.theta_start_thresh = 0.05   # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = 0.2     # threshold to be far enough into the plan to recompute it

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2 #orig was 0.2
        self.at_thresh = 0.1 #orig  was 0.02
        self.at_thresh_theta = 0.05
        self.near_delivery_thresh = 0.1 #orig was 0.2

        # trajectory smoothing
        self.spline_alpha = 0.12
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5 #orig was 0.5
        self.kpy = 0.5 #orig was 0.5
        self.kdx = 1.5 #orig was 1.5
        self.kdy = 1.5 #orig was 1.5

        # pose controller parameters
        self.k1 = 0.7
        self.k2 = 0.7
        self.k3 = 0.7

        # heading controller parameters
        self.kp_th = 2.0 #orig was 2.

        self.traj_controller = TrajectoryTracker(self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max)
        self.pose_controller = PoseController(self.k1, self.k2, self.k3, self.v_max, self.om_max)
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.nav_smoothed_path_pub = rospy.Publisher('/cmd_smoothed_path', Path, queue_size=10)
        self.nav_smoothed_path_rej_pub = rospy.Publisher('/cmd_smoothed_path_rejected', Path, queue_size=10)
        self.nav_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.inflated_occupancy_grid = rospy.Publisher('/inflated_occupancy_grid', OccupancyGrid, queue_size=10)

        self.trans_listener = tf.TransformListener()

        #self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)
        
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)
        rospy.Subscriber('/cmd_nav', Pose2D, self.cmd_nav_callback)
        rospy.Subscriber('/delivery_request',  String, self.delivery_callback)
        rospy.Subscriber('/detected_objects_list', DetectedObjectList, self.detected_obj_callback)
        rospy.Subscriber('/marker_topic_0', Marker, self.marker_callback)
        rospy.Subscriber('/marker_topic_1', Marker, self.marker_callback)
        rospy.Subscriber('/marker_topic_2', Marker, self.marker_callback)
        rospy.Subscriber('/marker_topic_3', Marker, self.marker_callback)
        rospy.Subscriber('/marker_topic_4', Marker, self.marker_callback)
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)

    def marker_callback(self, msg):
        self.marker_dict[msg.id] = (msg.pose.position.x, msg.pose.position.y) 

    def detected_obj_callback(self, msg):
        self.detected_objects_names = msg.objects
        self.detected_objects = msg.ob_msgs
        #update dictionary  as we get more objects
        if self.NUM_DELIVERY_ITEMS != len(self.detected_objects_names):
          self.NUM_DELIVERY_ITEMS = len(self.detected_objects_names)
          for i in range(len(self.detected_objects_names)):
            data = self.detected_objects[i]
            dist = data.distance
            #compute  location of food detected
            th_diff = 0.5*(wrapToPi(data.thetaleft) - wrapToPi(data.thetaright))
            th_center = wrapToPi(data.thetaleft) -  th_diff
            th_loc = self.theta + th_center
            food_loc_x  = self.x + dist*np.cos(th_loc)
            food_loc_y  = self.y + dist*np.sin(th_loc)
            LOCATIONS  =  (food_loc_x,food_loc_y,th_loc)
            self.objectname_markerLoc_dict[self.detected_objects_names[i]] = LOCATIONS #self.marker_dict[i]

    def delivery_callback(self, msg):
        if msg.data not in ['go to waypoints','home'] and not self.delivery_done:#self.detected_objects_names:
            self.delivery_done = True
            self.delivery_req_list = msg.data.split(',')
            first_item = self.delivery_req_list.pop(0)
            self.x_g, self.y_g, self.theta_g = self.objectname_markerLoc_dict[first_item]
            rospy.loginfo(first_item + " is at loc: " + str(self.x_g) + ", "+ str(self.y_g))
            self.replan()
        elif msg.data in ['go to waypoints']:
            #load first goal and go to navigation mode
            self.x_g, self.y_g, self.theta_g = waypoints.pose[self.vendor_ind]
            self.replan()
            #self.switch_mode(Mode.TRACK)            
        elif msg.data in ['home']:
            self.x_g = self.x_init
            self.y_g = self.y_init
            self.theta_g  = 4.71
            self.replan()
            self.waypoint_flag = False

    def dyn_cfg_callback(self, config, level):
        rospy.loginfo("Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}".format(**config))
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if data.x != self.x_g or data.y != self.y_g or data.theta != self.theta_g:

            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            rospy.loginfo("cmd_nav_callback goal: " + str(self.x_g)) 
            self.replan()

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x,msg.origin.position.y)

    def map_callback(self,msg):
        """
        receives new map info and updates the map
        msg.data is the map data, in row-major order, starting with (0,0).  Occupancy
        probabilities are in the range [0,100].  Unknown is -1.
        """
        self.map_probs = msg.data
        #we should get a int8 array, so we reshape into a 2D array
        map_probs2D = np.array(msg.data).reshape((msg.info.height,msg.info.width))
        #create a mask so that we don't touch -1 unknown values
        mask = map_probs2D<0
        #threshold so that anything below this level is set to 0
        map_probs2D_thresholded = (map_probs2D >= self.map_threshold) * 100
        #dilate the map so that we don't crash into walls
        map_probs2D_dilated = ndimage.binary_dilation(map_probs2D_thresholded,iterations=1).astype(np.int8) *100
        #add back unknown values into the map
        map_probs2D_dilated[mask] = -1
        #reshape back into original format
        inflated_OG = map_probs2D_dilated.reshape((msg.info.height*msg.info.width))

        #publish the inflated occupancy grid (for debugging)
        inflated_OG_msg = OccupancyGrid()
        inflated_OG_msg.info = msg.info
        inflated_OG_msg.data = inflated_OG
        self.inflated_occupancy_grid.publish(inflated_OG_msg)

        # if we've received the map metadata and have a way to update it:
        if self.map_width>0 and self.map_height>0 and len(self.map_probs)>0:
            self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  8,
                                                  inflated_OG)
            if self.x_g is not None and self.stop_flag != 1:
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan() # new map, need to replan

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """
        # distance of the stop sign
        dist = msg.distance
        self.stop_flag = 1
        # if close enough and in track mode, stop
        if dist > 0 and dist < self.stop_min_dist and self.mode == Mode.TRACK:
            self.init_stop_sign()

    def init_stop_sign(self):
         """ initiates a stop sign maneuver """
         rospy.loginfo("Stop Sign Detected")
         self.stop_sign_start = rospy.get_rostime()
         self.switch_mode(Mode.STOP)
    def has_stopped(self):
        """ checks if stop sign maneuver is over """
        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.stop_time)
    def stay_idle(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        self.nav_vel_pub.publish(vel_g_msg)
    def init_crossing(self):
        rospy.loginfo("Crossing")
        """ initiates an intersection crossing maneuver """

        self.cross_start = rospy.get_rostime()
        self.switch_mode(Mode.CROSS)

    def has_crossed(self):
        """ checks if crossing maneuver is over """

        return self.mode == Mode.CROSS and \
               rospy.get_rostime() - self.cross_start > rospy.Duration.from_sec(self.crossing_time)

    def near_mid_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return linalg.norm(np.array([self.x-self.x_mid, self.y-self.y_mid])) < self.near_thresh

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_thresh
    def near_delivery_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_delivery_thresh
    def at_home(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        return (linalg.norm(np.array([self.x-self.x_init, self.y-self.y_init])) < self.at_thresh and abs(wrapToPi(self.theta - 0.0)) < self.at_thresh_theta)


    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        return (linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.at_thresh and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta)

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh)
        
    def close_to_plan_start(self):
        return (abs(self.x - self.plan_start[0]) < self.start_pos_thresh and abs(self.y - self.plan_start[1]) < self.start_pos_thresh)

    def snap_to_grid(self, x):
        return (self.plan_resolution*round(x[0]/self.plan_resolution), self.plan_resolution*round(x[1]/self.plan_resolution))

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i,0]
            pose_st.pose.position.y = traj[i,1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def force_control(self):
        t = self.get_current_plan_time()
        V, om = self.traj_controller.compute_control(self.x, self.y, self.theta, t)
        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)        

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(self.x, self.y, self.theta, t)
        else:
            V = 0.
            om = 0.

        #print('V om ', V, om)

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime()-self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def wait(self):
        time0 = rospy.get_rostime()
        time  = rospy.get_rostime()
        while time-time0  <  rospy.Duration.from_sec(self.wait_time):
           time = rospy.get_rostime() 
           self.stay_idle()

    def force_move(self):
        #front of robot
        dist = 3*self.plan_resolution
        x_front = (self.x+dist*np.cos(self.theta), self.y+dist*np.sin(self.theta))
        x_back  = (self.x-dist*np.cos(self.theta), self.y-dist*np.sin(self.theta))
        if self.occupancy.is_free(x_front):
            rospy.loginfo("Navigator: Keep moving forwards")
            V  = 0.06
            om = 0.0
        elif self.occupancy.is_free(x_back):
            rospy.loginfo("Navigator: Keep moving backwards")
            V  = -0.06
            om = 0.0
        else:
            return -1
        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)
        return 1 

    def keep_moving(self):
        #front of robot
        dist = 3*self.plan_resolution
        x_front = (self.x+dist*np.cos(self.theta), self.y+dist*np.sin(self.theta))
        if self.occupancy.is_free(x_front):
            rospy.loginfo("Navigator: Keep moving forwards")
            V  = 0.06
            om = 0.0
        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel) 

    def check_neighbors(self,x_init,x_goal,dist):
        prob_list = []
        #forward, back, left, right, UL, UR, LL,LR
        neighbors = [self.snap_to_grid((self.x_g, self.y_g+dist)),
                     self.snap_to_grid((self.x_g, self.y_g-dist)),
                     self.snap_to_grid((self.x_g+dist, self.y_g)),
                     self.snap_to_grid((self.x_g-dist, self.y_g)),
                     self.snap_to_grid((self.x_g+dist, self.y_g+dist)),
                     self.snap_to_grid((self.x_g+dist, self.y_g-dist)),
                     self.snap_to_grid((self.x_g-dist, self.y_g+dist)),
                     self.snap_to_grid((self.x_g-dist, self.y_g-dist))]
        #check through all neighbors, appending valid paths that solve and are longer than a length of 4
        for i in range(len(neighbors)):
            rospy.loginfo("neighbor " + str(i))
            new_x_goal = neighbors[i]
            prob = AStar(self.state_min,self.state_max,x_init,new_x_goal,self.occupancy,self.plan_resolution)
            if prob.solve():
                if len(prob.path) > 4:
                    prob_list.append(prob)
        #go through all paths and pick shortest one. if none, then return none
        if len(prob_list) == 0:
            return None
        path_lengths = np.zeros(len(prob_list))
        for i in range(len(prob_list)):
            path_lengths[i] = len(prob_list[i].path)
        shortest_ind = np.argmin(path_lengths)
        return prob_list[shortest_ind]
            
    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo("Navigator: replanning canceled, waiting for occupancy map.")
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        #state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        #state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        #state_min = self.snap_to_grid((-0.05,-0.05))
        #state_max = self.snap_to_grid((self.plan_horizon,self.plan_horizon))

        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))

        rospy.loginfo('In replan, x_init,y_init,th_init:' + str(x_init) + ', '+str(self.th_init))
        rospy.loginfo('In replan, x_goal and th_g is:' + str(x_goal) + ', '+str(self.theta_g))

        problem = AStar(self.state_min,self.state_max,x_init,x_goal,self.occupancy,self.plan_resolution)

        rospy.loginfo("Navigator: computing navigation plan")
        success =  problem.solve()
        #if not success:
        #    rospy.loginfo("Planning failed")
        #    return
        #If not successful, keep trying to plan nearby
        ind = 0
        while not success:
            dist = ind*self.plan_resolution*4
            #self.intermediate_goal_flag = True
            rospy.loginfo("Planning failed searching distance " + str(dist) +" away from goal")
            #checks the neighbors and return the shortest path (making sure it's not too short)
            problem = self.check_neighbors(x_init,x_goal,dist) 
            if ind > 10:
                rospy.loginfo("too far. returning")
                return
            elif problem is None:
                rospy.loginfo("no viable path. keep checking neighbors further out")
                ind+=1
                continue
            else:
                rospy.loginfo("found a path, rerouting to "+str(problem.path[-1][0])+", " + str(problem.path[-1][1]))
                self.x_g = problem.path[-1][0]
                self.y_g = problem.path[-1][1]
                self.current_plan_start_time = rospy.get_rostime()
                #self.x_mid, self.y_mid = path[-1] 
                self.intermediate_goal_flag = False
                self.switch_mode(Mode.TRACK)
                break
        rospy.loginfo("Planning Succeeded")

        planned_path = problem.path
        

        # Check whether path is too short
        if len(planned_path) < 4:    
            rospy.loginfo("Path too short to track")
            #force to move and replan
            #time0 =  rospy.get_rostime()
            #time = rospy.get_rostime()
            #while time-time0 < rospy.Duration.from_sec(self.move_time):
            #    time = rospy.get_rostime()
            #    flag = self.force_move()
            #    if flag == -1:
            #         break
            self.switch_mode(Mode.TRACK)
            #self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        traj_new, t_new = compute_smoothed_traj(planned_path, self.v_des, self.spline_alpha, self.traj_dt)

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = self.current_plan_duration - self.get_current_plan_time()

            # Estimate duration of new trajectory
            th_init_new = traj_new[0,2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err/self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo("New plan rejected (longer duration than current plan)")
                self.publish_smoothed_path(traj_new, self.nav_smoothed_path_rej_pub)
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        #if self.intermediate_goal_flag:
        #    self.pose_controller.load_goal(mid_x_loc[0],mid_x_loc[1],self.theta_g)
        #else:
        #    self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)

        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0,2]
        self.heading_controller.load_goal(self.th_init)


        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print e
                pass

            #if  self.intermediate_goal_flag:
            #    self.wait()

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                pass
            elif self.mode == Mode.STOP:
                # At a stop sign
                while True:
                    self.stay_idle()
                    if self.has_stopped():
                        self.init_crossing()
                        self.stop_flag = 0
                        break
            elif self.mode == Mode.CROSS:
                # Crossing an intersection
                while True:
                    #self.keep_moving()
                    self.force_control()
                    if self.has_crossed():
                        #self.replan()
                        self.force_control()
                        break
                self.switch_mode(Mode.TRACK)

            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)

            elif self.mode == Mode.TRACK:
                if self.near_delivery_goal() and self.delivery_flag:
                    self.switch_mode(Mode.PARK)
                elif self.near_goal() and not self.delivery_flag:
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (rospy.get_rostime() - self.current_plan_start_time).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan() # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                if self.near_delivery_goal() and self.delivery_flag:
                    rospy.loginfo("Near delivery goal, forget about goal")
                    # forget about goal:
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    self.switch_mode(Mode.IDLE)
                    self.wait()
                    if len(self.delivery_req_list) != 0:
                        next_item = self.delivery_req_list.pop(0)
                        self.x_g, self.y_g, self.theta_g = self.objectname_markerLoc_dict[next_item]
                        self.replan()
                    else:
                        self.x_g = self.x_init
                        self.y_g = self.y_init
                        self.theta_g  = 0.0
                        self.home_flag = True
                        self.delivery_done = False
                        self.replan()
                    
                if self.at_goal() and self.waypoint_flag: #and not self.delivery_flag:
                    # forget about goal:
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    self.switch_mode(Mode.IDLE)

                    #when we are at goal, wait a few seconds and  move on to next waypoint
                    #if self.waypoint_flag:
                    self.wait()
                    if self.vendor_ind != NUM_WAYPOINTS:
                         self.x_g, self.y_g, self.theta_g = waypoints.pose[self.vendor_ind]
                         self.vendor_ind += 1
                         self.replan()
                    else:
                         self.x_g = self.x_init
                         self.y_g = self.y_init
                         self.theta_g  = 0.0
                         self.home_flag = True
                         self.replan()
                
                if self.home_flag and self.at_home():
                    # forget about goal:
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    self.switch_mode(Mode.IDLE)
                    if self.waypoint_flag:
                        self.waypoint_flag = False
                        self.delivery_flag = True
                        self.home_flag = False
     


            self.publish_control()

            rate.sleep()

if __name__ == '__main__':    
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
