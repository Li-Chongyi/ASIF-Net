# ASIF-Net

We provide the resutls of ASIF-Net in our paper. 
```
Google Drive: https://drive.google.com/open?id=15WlRLFSYG-mQ73DpUngaUqOnUwtz4PPc

Baidu Cloud: https://pan.baidu.com/s/1DVAwqe3n5JeUIuaAkzteYw  Password: byxj
```

We also provide our testing code. If you test our model, please download the pretrained model, unzip it, and put the checkpoint to checkpoint/coarse_224 folder.
```
Pretrained model download:

Google Drive: https://drive.google.com/file/d/1oTYUY9bKEFOrDyXMGTFFrgmFkHGUURM3/view?usp=sharing

Baidu Cloud: https://pan.baidu.com/s/1sQcb2o0of7sQXX3-D8ep5g  Password: 1234
```

# TensorFlow
```
TensorFlow implementation of ASIF-Net
```
## Requirements
```
Python 3
TensorFlow 1.x
```

## Data Preprocessing
```
1) normalize the depth maps (note that the foreground should have higher value than the background in our method);
2) resize the testing data to the size of 224*224;
3) put your rgb images to 'test_real' folder and your depth maps to 'depth_real' folder (paired rgb image and depth map should have same name)
```


### Test
```
python main_test.py

find the results in the 'test_real' folder with the same name as the input image + "_out".

You can use a script to resize the results back to the same size as the original RGB-D image,  or just use the results with a size of 224*224 for evaluations. We did not find much differences for the evaluation results.
```

## Bibtex


If you use the results and code, please cite our paper.
```
@article{ASIF-Net,
  title={{ASIF-Net}: Attention steered interweave fusion network for {RGBD} salient object detection},
  author={Li, Chongyi and Cong, Runmin and Kwong, Sam and Hou, Junhui and Fu, Huazhu and Zhu, Guopu and Zhang, Dingwen and Huang, Qingming },
  journal={IEEE Trans. Cybern.},
  volume={PP},
  number={99},
  pages={1-13},
  year={2020}
}

https://ieeexplore.ieee.org/document/8998588
```
## Contact
```
If you have any questions, please contact Chongyi Li at lichongyi25@gmail.com.
```
