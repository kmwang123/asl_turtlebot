import rospy
from std_msgs.msg import Float64

def pose_contrl_vars_msg(var):
    pubvar = rospy.Publisher('pose_contrl_var', Float64, queue_size=1) 
    #rospy.init_node('navigation_publisher')
    #rate = rospy.Rate(60) # 10h

    msg = Float64()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "pose_control"
    msg.data = Float64(var)
 
    while not rospy.is_shutdown():
            pubvar.publish(msg)
            rate.sleep() 
