import argparse
import time

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
# import gxipy as gx
import sys
import msvcrt
from ctypes import *
# sys.path.append("./MvImport")
from MvImport.MvCameraControl_class import *
import time

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name, cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

# class DaHengCamReader(object):
#     def __init__(self, cam_id):
#         self.device_manager = gx.DeviceManager()
#         dev_num, dev_info_list = self.device_manager.update_device_list()
#         if dev_num == 0:
#             print("Number of enumrated devices is 0")
#             return
#
#     def __iter__(self):
#         self.cam = self.device_manager.open_device_by_index(1)
#         self.cam.TriggerMode.set(gx.GxSwitchEntry.ON)
#         self.cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
#         self.cam.stream_on()
#         return self
#
#     def __next__(self):
#         # time.sleep(0.01)
#         self.cam.TriggerSoftware.send_command()
#         raw_image = self.cam.data_stream[0].get_image()
#         if raw_image is None:
#             print("Getting image failed.")
#             raise StopIteration
#             return None
#
#             # get RGB image from raw image
#         rgb_image = raw_image.convert("RGB")
#         # create numpy array with data from raw image
#         numpy_image = rgb_image.get_numpy_array()
#         cv_img = np.uint8(numpy_image)
#         img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
#         return img_bgr
#         # # show acquired image
#         # img = Image.fromarray(numpy_image, 'RGB')
#         # img.show()
#
#     def close(self):
#         self.cam.stream_off()
#         self.cam.close_device()

class HikCam(object):
    def __init__(self, active_way="getImagebuffer"):
        self.active_way = active_way
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            sys.exit()
        if deviceList.nDeviceNum == 0:
            print("Number of enumrated devices is 0")
            sys.exit()

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: %s" % strSerialNumber)

        # nConnectionNum = input("please input the number of the device to connect:")
        # if int(nConnectionNum) >= deviceList.nDeviceNum:
        #     print("intput error!")
        #     sys.exit()
        self.stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        return

    def __iter__(self):
        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(self.stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if self.stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # stBool = c_bool(False)
        # ret = self.cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
        # if ret != 0:
        #     print("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
        #
        # # ch:设置触发模式为off | en:Set trigger mode as off
        # ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        # if ret != 0:
        #     print("set trigger mode fail! ret[0x%x]" % ret)
        #     sys.exit()

        # ch:开始取流 | en:Start grab image
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()
        return self

    def image_show(self, image):
        image = cv2.resize(image, (1280, 1080), interpolation=cv2.INTER_AREA)
        cv2.imshow('fgmask', image)
        k = cv2.waitKey(1) & 0xff

    def image_resize(self, image):
        image_resize = cv2.resize(image, (1280, 1080), interpolation=cv2.INTER_AREA)
        return image_resize
    def image_control(self, data, stFrameInfo):
        image = None
        if stFrameInfo.enPixelType == 17301505:
            image = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
            image = self.image_resize(image=image)
        elif stFrameInfo.enPixelType == 17301514:
            data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
            image = cv2.cvtColor(data, cv2.COLOR_BAYER_GB2RGB)
            image =  self.image_resize(image=image)
        elif stFrameInfo.enPixelType == 35127316:
            data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
            image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            image =  self.image_resize(image=image)
        elif stFrameInfo.enPixelType == 34603039:
            data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
            image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_Y422)
            image =  self.image_resize(image=image)
        return image
    def __next__(self):
        image = None
        if self.active_way == "getImagebuffer":
            stOutFrame = MV_FRAME_OUT()
            memset(byref(stOutFrame), 0, sizeof(stOutFrame))
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if None != stOutFrame.pBufAddr and 0 == ret and stOutFrame.stFrameInfo.enPixelType == 17301505:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight,
                    stOutFrame.stFrameInfo.nFrameNum))
                pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
                cdll.msvcrt.memcpy(byref(pData), stOutFrame.pBufAddr,
                                   stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
                data = np.frombuffer(pData,
                                     count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight),
                                     dtype=np.uint8)
                image =  self.image_control(data=data, stFrameInfo=stOutFrame.stFrameInfo)
            elif None != stOutFrame.pBufAddr and 0 == ret and stOutFrame.stFrameInfo.enPixelType == 17301514:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight,
                    stOutFrame.stFrameInfo.nFrameNum))
                pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
                cdll.msvcrt.memcpy(byref(pData), stOutFrame.pBufAddr,
                                   stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
                data = np.frombuffer(pData,
                                     count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight),
                                     dtype=np.uint8)
                image =  self.image_control(data=data, stFrameInfo=stOutFrame.stFrameInfo)
            elif None != stOutFrame.pBufAddr and 0 == ret and stOutFrame.stFrameInfo.enPixelType == 35127316:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight,
                    stOutFrame.stFrameInfo.nFrameNum))
                pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3)()
                cdll.msvcrt.memcpy(byref(pData), stOutFrame.pBufAddr,
                                   stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3)
                data = np.frombuffer(pData,
                                     count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3),
                                     dtype=np.uint8)
                image =  self.image_control(data=data, stFrameInfo=stOutFrame.stFrameInfo)
            elif None != stOutFrame.pBufAddr and 0 == ret and stOutFrame.stFrameInfo.enPixelType == 34603039:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight,
                    stOutFrame.stFrameInfo.nFrameNum))
                pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 2)()
                cdll.msvcrt.memcpy(byref(pData), stOutFrame.pBufAddr,
                                   stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 2)
                data = np.frombuffer(pData,
                                     count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 2),
                                     dtype=np.uint8)
                image =  self.image_control(data=data, stFrameInfo=stOutFrame.stFrameInfo)
            else:
                print("no data[0x%x]" % ret)
            nRet = self.cam.MV_CC_FreeImageBuffer(stOutFrame)
        elif self.active_way == "getoneframetimeout":
            stParam = MVCC_INTVALUE_EX()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
            ret = self.cam.MV_CC_GetIntValueEx("PayloadSize", stParam)
            if ret != 0:
                print("get payload size fail! ret[0x%x]" % ret)
                sys.exit()
            nDataSize = stParam.nCurValue
            pData = (c_ubyte * nDataSize)()
            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

            ret = self.cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
            if ret == 0:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d] " % (
                    stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
                image = np.asarray(pData)
                image =  self.image_control(data=image, stFrameInfo=stFrameInfo)
            else:
                print("no data[0x%x]" % ret)
        return image

    def close(self):
        # ch:停止取流 | en:Stop grab image
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print("stop grabbing fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:关闭设备 | Close device
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            print("close deivce fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:销毁句柄 | Destroy handle
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            print("destroy handle fail! ret[0x%x]" % ret)
            sys.exit()


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img in image_provider:
        s_time = time.time()
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        # print('use time', time.time() - s_time)

        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--hik', type=str, default='', help='hik camera activate way')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '' and args.hik == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0
    # elif args.hik != '':
    #     frame_provider = HikCam(args.hik)

    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)


