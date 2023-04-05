# squeeze-ggcnn-master


**Note:** The program was modified by reference to Closing the Loop for Robotic Grasping.

This repository contains the implementation of the Generative Grasping Convolutional Neural Network (GG-CNN) from the paper:

**Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach**

*[Douglas Morrison](http://dougsm.com), [Peter Corke](http://petercorke.com), [Jürgen Leitner](http://juxi.net)*

Robotics: Science and Systems (RSS) 2018

[arXiv](https://arxiv.org/abs/1804.05172) | [Video](https://www.youtube.com/watch?v=7nOoxuGEcxA)


@inproceedings{morrison2018closing,
	title={{Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach}},
	author={Morrison, Douglas and Corke, Peter and Leitner, J\"urgen},
	booktitle={Proc.\ of Robotics: Science and Systems (RSS)},
	year={2018}
}


# Generative Grasping CNN (squeeze)

The squeeze is a lightweight, fully-convolutional network which predicts the quality and pose of antipodal grasps at every pixel in an input depth image.  The lightweight and single-pass generative nature of squeeze allows for fast execution and closed-loop control, enabling accurate grasping in dynamic environments where objects are moved during the grasp attempt.

## Installation
Operating system：Ubuntu MATE16.04
CPU：Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
GPU：Titan X
python版本：3.7.13
torch版本：1.10.1+cu111
torchvision版本：0.11.2++cu111

This code was developed with Python 3.7 on Ubuntu 16.04.  Python requirements can installed by:

```bash
pip install -r requirements.txt
```

## Datasets

Currently, both the [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php) and
[Jacquard Dataset](https://jacquard.liris.cnrs.fr/) are supported.

### Cornell Grasping Dataset

1. Download the and extract [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php). 
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`

### Jacquard Dataset

1. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/).


## Training

Training is done by the `train_ggcnn.py` script.  Run `train_ggcnn.py --help` to see a full list of options, such as dataset augmentation and validation options.

Some basic examples:

```bash
# Train Squeeze on Cornell Dataset
python train_ggcnn.py --description training_example --network squeezenet --dataset cornell --dataset-path <Path To Dataset>

# Train Squeeze on Jacquard Datset
python train_ggcnn.py --description training_example2 --network squeezenet --dataset jacquard --dataset-path <Path To Dataset>
```

```bash
python -m utils.dataset_processing.generate_cornell_depth  ../cornell_grasp_data

python train_ggcnn.py --description training_example --network squeezenet --dataset cornell --dataset-path ../cornell_grasp_data
```

Trained models are saved in `output/models` by default, with the validation score appended.

## Evaluation/Visualisation

Evaluation or visualisation of the trained networks are done using the `eval_ggcnn.py` script.  Run `eval_ggcnn.py --help` for a full set of options.

Important flags are:
* `--iou-eval` to evaluate using the IoU between grasping rectangles metric.
* `--jacquard-output` to generate output files in the format required for simulated testing against the Jacquard dataset.
* `--vis` to plot the network output and predicted grasping rectangles.

For example:

```bash
python eval_ggcnn.py --network <Path to Trained Network> --dataset jacquard --dataset-path <Path to Dataset> --jacquard-output --iou-eval
```

```bash
python eval_ggcnn.py  --network squeeze_weights_cornell/epoch_00_iou_0.61_statedict.pt --dataset cornell --dataset-path ../cornell_grasp_data --iou-eval   --vis
```

## Running on a Robot

Our ROS implementation for running the grasping system see [https://github.com/dougsm/mvp_grasp](https://github.com/dougsm/mvp_grasp).

The original implementation for running experiments on a Kinva Mico arm can be found in the repository [https://github.com/dougsm/ggcnn_kinova_grasping](https://github.com/dougsm/ggcnn_kinova_grasping).
