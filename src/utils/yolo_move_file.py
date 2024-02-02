import os
from glob import glob
from shutil import move


source_directory = "/home/hakertz-test/repos/6Dpose-Yolov7-Seg-AAE/yolov7/datasets/final/final_cem_obj_3"
destination_directory = "/home/hakertz-test/repos/6Dpose-Yolov7-Seg-AAE/yolov7/data"

for ext in ["png", "txt"]:
    last_dir = "images" if ext == "png" else "labels"
    for sub_dir in ["train", "val", "test"]:
        for src_path in glob("{}/{}/*.{}".format(source_directory, sub_dir, ext)):
            name_file = os.path.basename(src_path)
            dst_dir = "{}/{}/{}".format(destination_directory, sub_dir, last_dir)
            if os.path.isdir(dst_dir):
                dst = "{}/{}".format(dst_dir, name_file)
                move(src_path, dst)
        print("Finished {}".format(sub_dir))
print("Finished all")
