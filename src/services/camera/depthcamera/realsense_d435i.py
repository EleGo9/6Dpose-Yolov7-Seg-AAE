from src.services.camera.depthcamera.interface_depth_camera import IDepthCamera
import pyrealsense2 as rs
from time import sleep
from configparser import ConfigParser
import numpy as np
import cv2


class RealsenseD435I(IDepthCamera):
    def __init__(self, config_path, debug=False):
        self.config_path = config_path
        self.debug = debug

        self.config = None

        self.width = None
        self.height = None
        self.exposure = None
        self.auto_exposure = None
        self.gain = None
        self.brightness = None
        self.contrast = None
        self.gamma = None
        self.hue = None
        self.saturation = None
        self.sharpness = None

        self.rs = rs
        self.pipeline = None
        self.config = None

        self.started = None
        self.align = None
        self.frames = None
        self.depth_intrinsics = None

        self.initialize()

    def initialize(self):
        self.init_settings()
        self.init_camera()

    def init_settings(self):
        self.config = ConfigParser()
        self.config.read(self.config_path)

        self.width = eval(self.config.get("CAMERA", "width"))
        self.height = eval(self.config.get("CAMERA", "height"))

        self.auto_exposure = eval(self.config.get("CAMERA", "auto_exposure"))
        self.exposure = eval(self.config.get("CAMERA", "exposure"))
        self.gain = eval(self.config.get("CAMERA", "gain"))
        self.brightness = eval(self.config.get("CAMERA", "brightness"))
        self.contrast = eval(self.config.get("CAMERA", "contrast"))
        self.gamma = eval(self.config.get("CAMERA", "gamma"))
        self.hue = eval(self.config.get("CAMERA", "hue"))
        self.saturation = eval(self.config.get("CAMERA", "saturation"))
        self.sharpness = eval(self.config.get("CAMERA", "sharpness"))

    def init_camera(self):
        self.pipeline = self.rs.pipeline()

        config = self.rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)

        # start streaming
        self.started = self.pipeline.start(config)

        # align color and depth camera images
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        qs = self.started.get_device().query_sensors()[1]

        if not self.auto_exposure:
            qs.set_option(rs.option.exposure, self.exposure)
            qs.set_option(rs.option.gain, self.gain)
            qs.set_option(rs.option.brightness, self.brightness)
            qs.set_option(rs.option.contrast, self.contrast)
            qs.set_option(rs.option.gamma, self.gamma)
            qs.set_option(rs.option.hue, self.hue)
            qs.set_option(rs.option.saturation, self.saturation)
            qs.set_option(rs.option.sharpness, self.sharpness)
        else:
            sleep(2)

        if self.debug:
            print("exposure: {}".format(qs.get_option(rs.option.exposure)))
            print("gain: {}".format(qs.get_option(rs.option.gain)))
            print("brightness: {}".format(qs.get_option(rs.option.brightness)))
            print("contrast: {}".format(qs.get_option(rs.option.contrast)))
            print("gamma: {}".format(qs.get_option(rs.option.gamma)))
            print("hue: {}".format(qs.get_option(rs.option.hue)))
            print("saturation: {}".format(qs.get_option(rs.option.saturation)))
            print("sharpness: {}".format(qs.get_option(rs.option.sharpness)))

            rgb_profile = self.started.get_stream(rs.stream.color)
            rgb_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
            print("rgb_intrinsics: {}".format(rgb_intrinsics))
            depth_profile = self.started.get_stream(rs.stream.depth)
            depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            print("depth_intrinsics: {}".format(depth_intrinsics))

    def calibrate(self):
        pass

    def photo(self, rgb_path):
        # take RGB and depth frames together
        self.frames = self.pipeline.wait_for_frames()
        rgb_frame = self.frames.get_color_frame()

        rgb_image = np.asanyarray(rgb_frame.get_data())
        cv2.imwrite(rgb_path, rgb_image)

        return rgb_image

    def depth(self, depth_path, coordinates):
        # align rgb and depth frame
        aligned_frames = self.align.process(self.frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        self.depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        # apply filters to aligned depth frame
        aligned_depth_frame = rs.disparity_transform(True).process(aligned_depth_frame)
        aligned_depth_frame = rs.spatial_filter().process(aligned_depth_frame)
        aligned_depth_frame = rs.temporal_filter().process(aligned_depth_frame)
        aligned_depth_frame = rs.disparity_transform(False).process(aligned_depth_frame)
        aligned_depth_frame = rs.hole_filling_filter(0).process(aligned_depth_frame)

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        cv2.imwrite(depth_path, depth_image)
        # depth_colorized_image = np.asanyarray(rs.colorizer().colorize(aligned_depth_frame).get_data())

        return depth_image[coordinates[1], coordinates[0]].astype(float)

    def homography(self, direction, value, depth=0):
        if direction == "coordinates_to_mm":
            return rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, value, depth)
        elif direction == "mm_to_coordinates":
            return rs.rs2_project_point_to_pixel(self.depth_intrinsics, value)
        else:
            print('ERROR! Direction not allowed')

    def release(self):
        self.pipeline.stop()
