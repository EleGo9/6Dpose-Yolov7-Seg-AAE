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

#Evaluation
