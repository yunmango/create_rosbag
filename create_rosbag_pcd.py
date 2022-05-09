import glob

import numpy as np

from tqdm import tqdm

import pcl

import rospy
import rosbag
from sensor_msgs.msg import PointCloud2, PointField

def arr_to_pc2(np_pc, timestamp=None, frame_id='map'):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''    
    rviz_points = np.copy(np_pc)
    
    msg = PointCloud2()

    if timestamp is not None: 
        msg.header.stamp = timestamp
    else:
        msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    
    msg.height = 1
    msg.width = len(rviz_points)
    
    if np_pc.shape[1]==3:
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
    
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12*rviz_points.shape[0]
    elif np_pc.shape[1]==4:
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)]
    
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16*rviz_points.shape[0]
    elif np_pc.shape[1]==6:
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1)
        ]
        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = 24 * rviz_points.shape[0]

    else:
        None    
    
    msg.is_dense = int(np.isfinite(rviz_points).all())    
    
    msg.data = np.asarray(rviz_points, np.float32).tostring()

    return msg 

import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pcd_dir', help='pcd directory')
    args = parser.parse_args()

    # pcd_dir   = '/root/data/SOSLAB/ML2/posco/pointcloud_y03'
    pcd_dir   = args.pcd_dir
    file_name = pcd_dir.split('/')[-1]
    save_file   = pcd_dir + '/' + file_name + '.bag'

    rospy.init_node('create_rosbag')

    write_rosbag = rosbag.Bag(save_file, 'w')

    pcd_files   = sorted(glob.glob(pcd_dir+'/*.pcd'))
   
    frame_rate = 0.1     # 10hz = 0.1s
    start_time = rospy.get_time()

    for idx, pcd_file in enumerate(tqdm(pcd_files)):
        cur_time = rospy.Time.from_sec(start_time + frame_rate*idx)
        
        # load pcd
        pc = pcl.load(pcd_file)
        pc_arr = pc.to_array()

        # convert pcd to pc2 message
        msg_pc = arr_to_pc2(pc_arr)

        # write rosbag
        write_rosbag.write(topic='/ml/pointcloud', msg=msg_pc, t=cur_time)


    write_rosbag.close()