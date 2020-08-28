from __future__ import division
import numpy as np
from glob import glob
import os
# import scipy.misc
import cv2 as cv
# import sys
# sys.path.append('../../')
# from utils.misc import *

# 这里用的是data_odometry_color
class kitti_odom_loader(object):
    """KITTI odometry color读取类

    Attributes:
        seq_length: int类型的值。把seq_length个帧作为一组训练样本，这里要用奇数
        train_seqs: list，把列出来的KITTI序列作为训练数据
        test_seqs: list，把列出来的KITTI序列作为测试数据
    """
    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 seq_length=5):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        # 
        self.seq_length = seq_length
        self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.test_seqs = [9, 10]

        self.collect_test_frames()
        self.collect_train_frames()

    def collect_test_frames(self):
        """将测试数据（9,10序列）中图片的ID存入self.test_frames中"""
        self.test_frames = []
        for seq in self.test_seqs:
            seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
            img_dir = os.path.join(seq_dir, 'image_2')
            N = len(glob(img_dir + '/*.png'))
            for n in range(N):
                self.test_frames.append('%.2d %.6d' % (seq, n))
        self.num_test = len(self.test_frames)
        
    def collect_train_frames(self):from PIL import Image
        """将训练数据（0-8序列）中图片的ID存入self.train_frames中"""
        self.train_frames = []
        for seq in self.train_seqs:
            seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
            img_dir = os.path.join(seq_dir, 'image_2')
            N = len(glob(img_dir + '/*.png'))
            for n in range(N):
                self.train_frames.append('%.2d %.6d' % (seq, n))
        self.num_train = len(self.train_frames)

    def is_valid_sample(self, frames, tgt_idx):
        """验证抽取的样本是否合法。一个合法的样本不应该位于序列边缘"""
        N = len(frames)
        tgt_drive, _ = frames[tgt_idx].split(' ')
        half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_drive, _ = frames[min_src_idx].split(' ')
        max_src_drive, _ = frames[max_src_idx].split(' ')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
            return True
        return False

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        """读取一组图像数据，并缩放成指定大小(self.img_height, self.img_width)"""
        half_offset = int((seq_length - 1)/2)
        image_seq = []
        for o in range(-half_offset, half_offset+1):
            curr_idx = tgt_idx + o
            curr_drive, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image(curr_drive, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height/curr_img.size[0]
                zoom_x = self.img_width/curr_img.size[1]
            curr_img = cv.resize(curr_img, (self.img_width, self.img_height))
            # curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_example(self, frames, tgt_idx, load_pose=False):
        """读取并处理好一组图像数据
        
        Return:
            example = {
                'intrinsics' : 相机内参矩阵,
                'image_seq' : 5张一组的缩放后的图像,
                'folder_name' : 序列（sequence）号,
                'file_name' : 中间那张图片的id
            }
        """
        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        tgt_drive, tgt_frame_id = frames[tgt_idx].split(' ')
        intrinsics = self.load_intrinsics(tgt_drive, tgt_frame_id)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)        
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_drive
        example['file_name'] = tgt_frame_id
        if load_pose:
            pass
        return example

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx)
        return example

    # def load_frame(self, drive, frame_id):
    #     img = self.load_image(drive, frame_id)
    #     try:
    #         scale_x = np.float(self.img_width)/img.shape[1]
    #     except:
    #         print("KITTI loading error!")
    #         print("Drive = ", drive)
    #         print("frame_id = ", frame_id)
    #         raise
    #     scale_y = np.float(self.img_height)/img.shape[0]
    #     intrinsics = self.load_intrinsics(drive, frame_id)
    #     intrinsics = self.scale_intrinsics(intrinsics, scale_x, scale_y)
    #     img = self.crop_resize(img)
    #     return img, intrinsics

    def load_image(self, drive, frame_id):
        img_file = os.path.join(self.dataset_dir, 'sequences', '%s/image_2/%s.png' % (drive, frame_id))
        img = cv.imread(img_file)
        cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # img = scipy.misc.imread(img_file)
        return img

    def load_intrinsics(self, drive, frame_id):
        """读取3x3的内参矩阵"""
        calib_file = os.path.join(self.dataset_dir, 'sequences', '%s/calib.txt' % drive)
        proj_c2p, _ = self.read_calib_file(calib_file)
        intrinsics = proj_c2p[:3, :3]
        return intrinsics

    # def load_gt_odom(self, drive, tgt_idx, src_idx):
    #     pose_file = os.path.join(self.dataset_dir, 'poses', '%s.txt' % drive)
    #     with open(pose_file, 'r') as f:
    #         poses = f.readlines()
    #     filler = np.array([0, 0, 0, 1]).reshape((1,4))
    #     tgt_pose = np.array(poses[int(tgt_idx)][:-1].split(' ')).astype(np.float32).reshape(3,4)
    #     tgt_pose = np.concatenate((tgt_pose, filler), axis=0)
    #     src_pose = np.array(poses[int(src_idx)][:-1].split(' ')).astype(np.float32).reshape(3,4)
    #     src_pose = np.concatenate((src_pose, filler), axis=0)
    #     rel_pose = np.dot(np.linalg.inv(src_pose), tgt_pose)
    #     rel_6DOF = pose_mat_to_6dof(rel_pose)
    #     return rel_6DOF

    def read_calib_file(self, filepath, cid=2):
        """Read in a calibration file and parse into a dictionary.
        
        calib.txt中存放了四个相机的参数，参考https://zhuanlan.zhihu.com/p/99114433
        """
        with open(filepath, 'r') as f:
            C = f.readlines()
        def parseLine(L, shape):
            data = L.split()
            data = np.array(data[1:]).reshape(shape).astype(np.float32)
            return data
        proj_c2p = parseLine(C[cid], shape=(3,4))
        proj_v2c = parseLine(C[-1], shape=(3,4)) #这里似乎是想拿Tr_velo_to_cam，后面用不到这个值
        filler = np.array([0, 0, 0, 1]).reshape((1,4))
        proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
        return proj_c2p, proj_v2c

    def scale_intrinsics(self,mat, sx, sy):
        """根据对图片尺寸的缩放修改内参矩阵"""
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out


