# Intel® Movidius™ NCS examples

## MNIST with Keras

Train a simple CNN for MNIST:

```
python train-mnist.py
```

Convert Keras model to Tensorflow model

```
python load-mnist.py
```

Check, Compile, Profile

```
mvNCCheck TF_Model/tf_model.meta -in=conv2d_1_input -on=dense_2/Softmax
mvNCProfile TF_Model/tf_model.meta -in=conv2d_1_input -on=dense_2/Softma
mvNCCompile TF_Model/tf_model.meta -in=conv2d_1_input -on=dense_2/Softma
```

## VGG16 with Keras

Save the VGG16 model weights and config

```
python load-vgg16.py
```

Check, Compile, Profile

```
mvNCCheck TF_Model/vgg16.meta -in=input_1 -on=predictions/Softmax -s 12
mvNCCompile TF_Model/vgg16.meta -in=input_1 -on=predictions/Softmax -s 12
```

```
mvNCProfile TF_Model/vgg16.meta -in=input_1 -on=predictions/Softmax -s 12
```

Output:

```
...
Detailed Per Layer Profile
                                                                              Bandwidth   time
#    Name                                                               MFLOPs  (MB/s)    (ms)
==============================================================================================
0    block1_conv1/Relu                                                   173.4   303.2   8.537
1    block1_conv2/Relu                                                  3699.4   664.3  83.092
2    block1_pool/MaxPool                                                   3.2   831.3   7.369
3    block2_conv1/Relu                                                  1849.7   419.7  33.173
4    block2_conv2/Relu                                                  3699.4   474.1  58.732
5    block2_pool/MaxPool                                                   1.6   922.9   3.318
6    block3_conv1/Relu                                                  1849.7   172.9  43.128
7    block3_conv2/Relu                                                  3699.4   179.5  83.084
8    block3_conv3/Relu                                                  3699.4   178.4  83.578
9    block3_pool/MaxPool                                                   0.8   932.5   1.643
10   block4_conv1/Relu                                                  1849.7   136.1  41.908
11   block4_conv2/Relu                                                  3699.4   169.8  67.143
12   block4_conv3/Relu                                                  3699.4   170.7  66.792
13   block4_pool/MaxPool                                                   0.4   923.5   0.830
14   block5_conv1/Relu                                                   924.8   316.8  19.669
15   block5_conv2/Relu                                                   924.8   310.3  20.080
16   block5_conv3/Relu                                                   924.8   315.7  19.739
17   block5_pool/MaxPool                                                   0.1   878.8   0.219
18   fc1/Relu                                                            205.5  2159.0  90.811
19   fc2/Relu                                                             33.6  2139.0  14.967
20   predictions/BiasAdd                                                   8.2  2640.5   2.962
21   predictions/Softmax                                                   0.0    15.1   0.253
----------------------------------------------------------------------------------------------
                                                  Total inference time                  751.03
----------------------------------------------------------------------------------------------
Generating Profile Report 'output_report.html'...
```


## Reference

+ [ardamavi/Intel-Movidius-NCS-Keras](https://github.com/ardamavi/Intel-Movidius-NCS-Keras)
