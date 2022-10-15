# mmdrive: mmWave Sensing for Live Monitoring and On-Device Inference of Dangerous Driving

In this work we explore the feasibility of purely using mmWave radars to detect dangerous driving behaviors. We then develop a novel Fused-CNN model to detect dangerous driving instances from regular driving and classify 9 different
dangerous driving actions. Through extensive  experiments with 5 volunteer drivers in real driving environments, we observe that our system can distinguish dangerous driving actions with an
average accuracy of 97(±2)%. 

## Installation:

To install use the following commands.
```bash
git clone <annonymous-url>
pip install -r requirements.txt
```

## Directory Structure


```
mmdrive
└── models
    └── fused_cnn.py
    └── rf.py
    └── vgg_16.py
└── acoustic_fmcw
└── results
└── dataset
    └── dataset_pub.pkl
└── mmwave_demo_visualizer
```