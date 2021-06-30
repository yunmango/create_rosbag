import glob

import numpy as np

import cv2

import rospy
import rosbag
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image


def get_bin_points(bin_file):  
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points

def get_txt_points(txt_file):
    points = np.loadtxt(txt_file, dtype=np.float32)
    return points

def get_img(img_file):
    img = cv2.imread(img_file)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


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




if __name__=='__main__':
    pc_raw_xyzi_dir   = '/root/data/SOSLAB/ML/210618/original/raw/lidar/lidar'
    pc_raw_xyzrgb_dir = '/root/data/SOSLAB/ML/210618/Yun/xyzrgb_txt/raw/txt'
    pc_xyzi_dir       = '/root/data/SOSLAB/ML/210618/original/result/lidar/lidar'
    pc_xyzrgb_dir     = '/root/data/SOSLAB/ML/210618/Yun/xyzrgb_txt/result/txt'
    img_dir           = '/root/data/SOSLAB/ML/210618/Yun/xyzrgb_txt/result/png'

    rospy.init_node('create_rosbag')
    bridge = CvBridge()

    save_path = '/root/data/SOSLAB/ML/210618/ml.bag'
    write_rosbag = rosbag.Bag(save_path, 'w')

    list_bin_pc_raw_xyzi   = sorted(glob.glob(pc_raw_xyzi_dir+'/*.bin'))
    list_txt_pc_raw_xyzrgb = sorted(glob.glob(pc_raw_xyzrgb_dir+'/*.txt'))
    list_bin_pc_xyzi       = sorted(glob.glob(pc_xyzi_dir+'/*.bin'))
    list_txt_pc_xyzrgb     = sorted(glob.glob(pc_xyzrgb_dir+'/*.txt'))
    list_img               = sorted(glob.glob(img_dir+'/*.png'))

    print(len(list_bin_pc_raw_xyzi))
    print(len(list_txt_pc_raw_xyzrgb))
    print(len(list_bin_pc_xyzi))
    print(len(list_txt_pc_xyzrgb))
    print(len(list_img))


    frame_rate = 0.1     # 10hz = 0.1s
    start_time = rospy.get_time()

    for idx in range(len(list_img)): 
        print(idx)
        
        print(list_bin_pc_raw_xyzi[idx])
        print(list_txt_pc_raw_xyzrgb[idx])
        print(list_bin_pc_xyzi[idx])
        print(list_txt_pc_xyzrgb[idx])
        print(list_img[idx])
        print('-----------------------------------')
        
        # get points
        pc_raw_xyzi = get_bin_points(list_bin_pc_raw_xyzi[idx])
        pc_raw_xyzrgb = get_txt_points(list_txt_pc_raw_xyzrgb[idx])
        pc_raw_xyzrgb[:,3:6] = pc_raw_xyzrgb[:,3:6]/255.0
        pc_xyzi = get_bin_points(list_bin_pc_xyzi[idx])
        pc_xyzrgb = get_txt_points(list_txt_pc_xyzrgb[idx])
        pc_xyzrgb[:,3:6] = pc_xyzrgb[:,3:6]/255.0

        # get image
        img = get_img(list_img[idx])

        cur_time = rospy.Time.from_sec(start_time + frame_rate*idx)
        
        msg_pc_raw_xyzi   = arr_to_pc2(pc_raw_xyzi)
        msg_pc_raw_xyzrgb = arr_to_pc2(pc_raw_xyzrgb)
        msg_pc_xyzi       = arr_to_pc2(pc_xyzi)
        msg_pc_xyzrgb     = arr_to_pc2(pc_xyzrgb)
        msg_img           = bridge.cv2_to_imgmsg((img))
    
        write_rosbag.write(topic='/soslab_ml/pc_raw_xyzi', msg=msg_pc_raw_xyzi, t=cur_time)
        write_rosbag.write(topic='/soslab_ml/pc_raw_xyzrgb', msg=msg_pc_raw_xyzrgb, t=cur_time)
        write_rosbag.write(topic='/soslab_ml/pc_xyzi', msg=msg_pc_xyzi, t=cur_time)
        write_rosbag.write(topic='/soslab_ml/pc_xyzrgb', msg=msg_pc_xyzrgb, t=cur_time)
        write_rosbag.write(topic='/soslab_ml/img_ambient', msg=msg_img, t=cur_time)

    
    write_rosbag.close()


