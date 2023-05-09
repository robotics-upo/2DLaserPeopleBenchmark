import rosbag
import numpy as np

from rospy import Time
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

#from tf.transformations import euler_from_quaternion # ROS TF, not TensorFlow!

FROG_PATH = '/mnt/data/work/datasets/frog_all'

BAG_TOPIC_SCAN = '/scanfront'
BAG_TOPIC_ODOM = '/pose'

BAGMAP = {
	'10-31': '2014-04-30-10-31-26',
	'11-36': '2014-04-29-11-36-22', # trainval1
	'12-43': '2014-04-30-12-43-38', # trainval2
	'14-57': '2014-04-29-14-57-50',
	'15-53': '2014-04-28-15-53-18',
	'16-41': '2014-04-29-16-41-49', # test
}

def get_z_angle(q : Quaternion):
	y = 2. * (q.w * q.z + q.x * q.y)
	x = 1. - 2. * (q.y * q.y + q.z * q.z)
	return np.arctan2(y, x)

#def get_quat(q):
#	return np.array([ q.x, q.y, q.z, q.w ])

for bag_id, bag_file in BAGMAP.items():
	print(bag_id)

	odom_ts   = []
	odom_data = []

	scan_ts   = []
	scan_data = []

	with rosbag.Bag(f'{FROG_PATH}/UPO_pioneer_sensors_{bag_file}.bag', 'r') as f:
		for topicname, msg, ts in f.read_messages(topics=[BAG_TOPIC_SCAN, BAG_TOPIC_ODOM]):
			ts : Time

			if topicname==BAG_TOPIC_ODOM:
				msg : Odometry

				odom_pose = msg.pose.pose
				odom_position = odom_pose.position
				odom_orientation = odom_pose.orientation
				#print(euler_from_quaternion(get_quat(odom_orientation)))

				odom_ts.append(ts.to_sec())
				odom_data.append([ odom_position.x, odom_position.y, get_z_angle(odom_orientation) ])
			elif topicname==BAG_TOPIC_SCAN:
				msg : LaserScan

				scan_ts.append(ts.to_sec())
				scan_data.append(msg.ranges)

	odom_ts   = np.array(odom_ts,   dtype=np.float64)
	odom_data = np.array(odom_data, dtype=np.float32)
	scan_ts   = np.array(scan_ts,   dtype=np.float64)
	scan_data = np.array(scan_data, dtype=np.float32)

	np.savez_compressed(f'{FROG_PATH}/interm/frog_{bag_id}_odom.npz',
		ts   = odom_ts,
		data = odom_data,
	)

	print('  ',len(odom_ts),'odometry samples exported')

	np.savez_compressed(f'{FROG_PATH}/interm/frog_{bag_id}_scans.npz',
		ts   = scan_ts,
		data = scan_data,
	)

	print('  ',len(scan_ts),'scans exported')
