from time import strftime
import cv2


class Results:
    def __init__(self, dir_path, save=True):
        self.dir_path = dir_path
        self.save = save

        self.path = ""

        self.initialize()

    def initialize(self):
        pass

    def set_timestamp(self):
        self.path = self.dir_path + strftime("%y%m%d_%H%M%S") + "_{}"

    def save_image(self, image):
        if self.save:
            cv2.imwrite(self.path.format("image.png"), image)

    def save_detection(self, labels, boxes):
        if self.save:
            with open(self.path.format("detection.txt"), "w") as f:
                print(labels, boxes)
                for l, b in zip(labels, boxes):
                    print(l, b)
                    f.write("{} {} {} {} {}\n".format(l, *b))

    def save_segmentation(self, masks, segmentation_image):
        if self.save:
            for i, m in enumerate(masks):
                mask_image = m.cpu().numpy()*255
                cv2.imwrite(self.path.format("mask_{}.png".format(i)), mask_image)

            cv2.imwrite(self.path.format("segmentation_image.png"), segmentation_image)

    def save_processing(self, masked_image):
        cv2.imwrite(self.path.format("masked_image.png"), masked_image)

    def save_6d_pose_estimation(self, rotation_matrices, pose_estimation_image):
        if self.save:
            with open(self.path.format("6d_pose_estimation.txt"), "w") as f:
                for rm in rotation_matrices:
                    for r in rm:
                        f.write("{} {} {} {}\n".format(*r))
                    f.write("\n")

            cv2.imwrite(self.path.format("6d_pose_estimation_image.png"), pose_estimation_image)

    def save_picking_point(self, rotation_matrices, xyz_mm, picking_points_image):
        if self.save:
            with open(self.path.format("picking_points.txt"), "w") as f:
                for rm in rotation_matrices:
                    for r in rm:
                        f.write("{} {} {} {}\n".format(*r))
                f.write("{} {} {}\n".format(*xyz_mm))

            cv2.imwrite(self.path.format("picking_points_image.png"), picking_points_image)

    def save_correct_picking(self, correct_picking):
        if self.save:
            with open(self.path.format("correct_picking.txt"), "w") as f:
                f.write("{}".format(correct_picking))
