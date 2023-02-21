# 6Dpose-Yolov7-Seg-AAE
New pipeline proposed based on segmentation (with Yolov7) + pose estimation (with Augmented Autoencoder) from RGB monocular
# Env Creation
AAE requirements
```
sudo apt-get install libglfw3-dev libglfw3
sudo apt-get install libassimp-dev
conda env create -f aae_py37_tf26.yml
conda activate aae_seg2
pip install --pre --upgrade PyOpenGL PyOpenGL_accelerate
pip install cython
pip install cyglfw3
pip install pyassimp==3.3
pip install imgaug
pip install progressbar
```

Then enter with cd in the AAE folder and install auto_pose:
```
pip install .
```

For yolov7 requirements:
```
pip install -r yolov7_requirements.txt
```

# Evaluation
```
export PYOPENGL_PLATFORM='egl'
export AE_WORKSPACE_PATH=/path/to/ae_workspace/
python src/demo_yolov7_aae.py -f /path/to/the/data/folder -test_config /path/to/config/file.cgf -save_res /path/where/you/want/results
```
Args for the demo script are:
- -f or --file_path indicate path to the folder where images to predict are placed
- --seg_yes is an option of adding or removing segmentation in the input of aae (True with segmentation, False without)
- -d is the path to depth map folder (not used for the moment)
- -i is the path for a bag file (not used for the moment)
- -v is the path for a video file (not used for the moment)
- -r is the path for a realsense input video (not used for the moment)
- -test_config is the path for cfg_eval file
- -vis is 
- -debugvis if True the code show images step by step, otherwise it doesn't show anything



