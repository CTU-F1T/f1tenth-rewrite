"""Substitutes every LiDAR measurement with a circle obstacle.

Simple recognition, example for this layer.

 -------    /scan     ------------      /obstacles     ---------
| LiDAR | ---------> | substitute | ----------------> | Planner |
 -------  LaserData   ------------   ObstaclesStamped  ---------
"""
######################
# Imports & Globals
######################

# Handling arguments
import sys

# ROS python package
import rospy

# Computation engine
import math

# Timing
import time

# Delay and timing measurement
from rosmeasure.utils import TimeMeasurer, DelayMeasurer


# Message types
# CircleObstacle
from obstacle_msgs.msg import CircleObstacle
# geometry_msgs/Point center
# float64 radius
# geometry_msgs/Vector3 velocity        # applied to center

# Header
from std_msgs.msg import Header
# uint32 seq                            # sequence ID
# time stamp                            # timestamp
# string frame_id                       # Frame this data is associated with

# LaserScan
from sensor_msgs.msg import LaserScan
# std_msgs/Header header
# float32 angle_min         # start angle of the scan [rad]
# float32 angle_max         # end angle of the scan [rad]
# float32 angle_increment   # angular distance between measurements [rad]
# float32 time_increment    # time between measurements [seconds]
# float32 scan_time         # time between scans [seconds]
# float32 range_min         # minimum range value [m]
# float32 range_max         # maximum range value [m]
# float32[] ranges          # range data [m] (Note: values < range_min or > range_max should be discarded)
# float32[] intensities     # intensity data [device-specific units]

# Obstacles
from obstacle_msgs.msg import Obstacles
# obstacle_msgs/CircleObstacle[] circles
# ... rest is omitted as it is not used here.

# ObstaclesStamped
from obstacle_msgs.msg import ObstaclesStamped
# std_msgs/Header header
# obstacle_msgs/Obstacles obstacles

# Point
from geometry_msgs.msg import Point
# float64 x
# float64 y
# float64 z

# Vector3
from geometry_msgs.msg import Vector3
# float64 x
# float64 y
# float64 z


# Publishers
pub = rospy.Publisher("/obstacles", ObstaclesStamped, queue_size = 1)


# Measurers
scan_m = TimeMeasurer("callback_scan", "ms")
scan_d = DelayMeasurer("scan_delay", "ms")


######################
# Utility functions
######################

def polar_to_point(distance, angle):
    """Converts polar coordinates of a 2D point into cartesian coordinates.

    Arguments:
    distance -- distance to the point [m]
    angle -- angle between x_axis and point [rad]

    Returns:
    point -- point in cartesian coordinates, geometry_msgs.msg/Point
    """
    point = Point()

    point.x = math.cos(angle) * distance
    point.y = math.sin(angle) * distance
    point.z = 0.0

    return point


######################
# Callbacks
######################

def callback_scan(data):
    """Converts each single measurement into a CircleObstacle and publishes them.

    Arguments:
    data -- structure received on a /scan topic, defined by obstacle_msgs.msg/ObstaclesStamped
    """

    scan_m.start()

    msg = ObstaclesStamped()

    msg.header = data.header

    msg.obstacles = Obstacles()

    msg.obstacles.circles = list()

    angle = data.angle_min - data.angle_increment

    for m in data.ranges:
        angle += data.angle_increment

        if math.isnan(m) or math.isinf(m) or m > data.range_max or m < data.range_min:
            continue

        obs = CircleObstacle()

        obs.center = polar_to_point(m, angle)
        obs.radius = 0.01

        msg.obstacles.circles.append(obs)

    scan_m.end()
    scan_m.summary()

    scan_d.delay(data.header)
    scan_d.summary()

    pub.publish(msg)


######################
# Functions
######################

def start_node(anonymous = False):
    """Starts a ROS node, registers the callbacks."""
    # Only one should be running
    # But for simulation we allow multiple nodes
    rospy.init_node("obstacle_substitution", anonymous = anonymous)

    # Register a callback
    rospy.Subscriber("/scan", LaserScan, callback_scan)

    # Function spin() simply keeps python from exiting until this node is stopped.
    rospy.spin()


if __name__ == '__main__':
    args = [ a.lower() for a in rospy.myargv(argv = sys.argv) ]

    # Slightly ugly, but efficient solution
    if "anonymous=true" in args:
        start_node(True)
    else:
        start_node(False)
