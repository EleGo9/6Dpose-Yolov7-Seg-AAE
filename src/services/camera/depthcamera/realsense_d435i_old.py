from src.services.camera.depthcamera.interface_depth_camera import IDepthCamera
import pyrealsense2 as rs
import sys
import time
import numpy as np
import cv2


class RealsenseD435I(IDepthCamera):
    def __init__(self, resolution, enable_logging):
        self.enable_logging = enable_logging
        self.pipeline = None
        self.config = None
        self.resolution = resolution  # (640, 480)
        self.started = None
        self.align = None
        self.frames = None
        self.depth_intrinsics = None

        self.initialize()

    def initialize(self):
        if self.enable_logging:
            sys.stdout.write('Initialize\n')

        # Start of the neuralnetworks
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Pipeline configuration
        self.config.enable_stream(rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.resolution[0], self.resolution[1], rs.format.bgr8, 30)

        # Start of the streaming
        self.started = self.pipeline.start(self.config)

        # Creating an alignment object
        # rs.align premises to obtain an alignment between color and depth camera images
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        qs = self.started.get_device().query_sensors()[1]
        qs.set_option(rs.option.exposure, 85)
        qs.set_option(rs.option.gain, 40)
        qs.set_option(rs.option.brightness, 30)
        qs.set_option(rs.option.contrast, 60)
        qs.set_option(rs.option.gamma, 300)
        qs.set_option(rs.option.hue, 0)
        qs.set_option(rs.option.saturation, 30)
        qs.set_option(rs.option.sharpness, 80)

        print(qs.get_option(rs.option.exposure))
        print(qs.get_option(rs.option.gain))
        print(qs.get_option(rs.option.brightness))
        print(qs.get_option(rs.option.contrast))
        print(qs.get_option(rs.option.gamma))
        print(qs.get_option(rs.option.hue))
        print(qs.get_option(rs.option.saturation))
        print(qs.get_option(rs.option.sharpness))

        profile = self.started.get_stream(rs.stream.color)
        intr = profile.as_video_stream_profile().get_intrinsics()
        print(intr)
        profile = self.started.get_stream(rs.stream.depth)
        intr = profile.as_video_stream_profile().get_intrinsics()
        print(intr)
        # exit()

        # Auto-exposure stabilization time
        time.sleep(2)

    def calibrate(self):
        global new_calib
        if self.enable_logging:
            sys.stdout.write('Calibrate\n')

        # Creating calibration object
        calib_dev = rs.auto_calibrated_device(self.started.get_device())

        # Calibration options
        while True:
            try:
                operation = input("Please select what the operation you want to do\nc - on chip calibration\n"
                                  "t - tare calibration\ng - get the active calibration\nw - write new calibration\n"
                                  "e - exit\n")

                if operation == 'c':
                    print("Starting on chip calibration")
                    new_calib, health = calib_dev.run_on_chip_calibration(file_cnt, on_chip_calib_cb, 5000)
                    print("Calibration completed")
                    print("health factor = ", health)

                if operation == 't':
                    print("Starting tare calibration")
                    ground_truth = float(input("Please enter ground truth in mm\n"))
                    new_calib, health = calib_dev.run_tare_calibration(ground_truth, file_cnt, on_chip_calib_cb, 5000)
                    print("Calibration completed")
                    print("health factor = ", health)

                if operation == 'g':
                    calib = calib_dev.get_calibration_table()
                    print("Calibration", calib)

                if operation == 'w':
                    print("Writing the new calibration")
                    calib_dev.set_calibration_table(new_calib)
                    calib_dev.write_calibration()

                if operation == 'e':
                    break

                print("Done\n")
            except Exception as e:
                print(e)

    def photo(self, image_name):
        if self.enable_logging:
            print('Photo')

        # Taking RGB and depth frames together
        self.frames = self.pipeline.wait_for_frames()
        color_frame = self.frames.get_color_frame()

        # Converting frames as numpy arrays to passing them to OpenCV
        color_image = np.asanyarray(color_frame.get_data())

        # Errors managing
        if not color_frame:
            sys.stderr.write('ERROR! Something has gone wrong in taking colour frame.\n')
        else:
            # Writing color frame
            # cv2.imwrite("src/services/webapp/static/img/camera.png", color_image)
            cv2.imwrite(image_name, color_image)
        return color_image

    def depth(self, image_name, coordinates):
        if self.enable_logging:
            sys.stdout.write('Depth\n')

        # Aligning depth frame to colour frame
        aligned_frames = self.align.process(self.frames)
        # aligned_depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()

        # Storing the alignment intrinsics values
        self.depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        # Applying filters to aligned depth frame
        aligned_depth_frame = rs.disparity_transform(True).process(aligned_depth_frame)
        aligned_depth_frame = rs.spatial_filter().process(aligned_depth_frame)
        aligned_depth_frame = rs.temporal_filter().process(aligned_depth_frame)
        aligned_depth_frame = rs.disparity_transform(False).process(aligned_depth_frame)
        aligned_depth_frame = rs.hole_filling_filter(0).process(aligned_depth_frame)

        # Frame from taking deepness
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # Frame to store colorized
        depth_colorized_image = np.asanyarray(rs.colorizer().colorize(aligned_depth_frame).get_data())

        # Errors managing
        if np.isnan(depth_image).any():
            sys.stderr.write('ERROR! Something has gone wrong in taking depth frame.\n')
        else:
            # Writing depth colorized frame
            #cv2.circle(depth_colorized_image, coordinates, 4, (0, 0, 255))
            #cv2.imwrite(image_name, depth_colorized_image)
            #inv_depth_image = depth_image
            cv2.imwrite(image_name, depth_image)

        return depth_image[coordinates[1], coordinates[0]].astype(float)

    def homography(self, direction, value, depth=0):
        if self.enable_logging:
            sys.stdout.write('Homography\n')

        # controlling requested direction
        if direction == "coordinates_to_mm":
            return rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, value, depth)
        elif direction == "mm_to_coordinates":
            return rs.rs2_project_point_to_pixel(self.depth_intrinsics, value)
        else:
            sys.stderr.write('ERROR! Direction not allowed.\n')

    def release(self):
        if self.enable_logging:
            sys.stdout.write('Release\n')

        # Chiusura della connesione con la camera
        self.pipeline.stop()


if __name__ == "__main__":
    print("starting camera")
    depthCamera = RealsenseD435I((640, 480), False)
    depthCamera.initialize()

    print("taking photo")
    # depthCamera.photo("src/processes/tests/images/rgb.png")
    # depthCamera.depth("src/processes/tests/images/depth.png", (0, 0))
    depthCamera.photo("/home/hakertz-test/repos/FFB6D/ffb6d/datasets/linemod/Linemod_preprocessed/data/10/rgb/0016.png")
    depthCamera.depth("/home/hakertz-test/repos/FFB6D/ffb6d/datasets/linemod/Linemod_preprocessed/data/10/depth/0016.png", (0, 0))


    depthCamera.release()
    print("finished")
