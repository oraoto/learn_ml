# Intel® Movidius™ NCS examples

## MNIST with Keras

Train a simple CNN for MNIST:

```
$ python train-mnist.py
```

Convert Keras model to Tensorflow model

```
$ python convert-mnist.py
```

Check, Compile, Profile

```
$ mvNCCheck TF_Model/tf_model.meta -in=conv2d_1_input -on=dense_2/Softmax
$ mvNCCompile TF_Model/tf_model.meta -in=conv2d_1_input -on=dense_2/Softmax
$ mvNCProfile TF_Model/tf_model.meta -in=conv2d_1_input -on=dense_2/Softmax
```

Do prediction on a random image:

```
$ python predict-mnist.py
```

## VGG16 with Keras

Convert the VGG16 model to TensorFlow model

```
$ python convert-vgg16.py
```

Check, Compile, Profile

```
$ mvNCCheck TF_Model/vgg16.meta -in=input_1 -on=predictions/Softmax -s 12
$ mvNCCompile TF_Model/vgg16.meta -in=input_1 -on=predictions/Softmax -s 12
$ mvNCProfile TF_Model/vgg16.meta -in=input_1 -on=predictions/Softmax -s 12
```

Profile output:

```
Detailed Per Layer Profile
                                                 Bandwidth   time
#    Name                                  MFLOPs  (MB/s)    (ms)
=================================================================
0    block1_conv1/Relu                      173.4   304.1   8.512
1    block1_conv2/Relu                     3699.4   664.6  83.057
2    block1_pool/MaxPool                      3.2   831.7   7.365
3    block2_conv1/Relu                     1849.7   419.9  33.161
4    block2_conv2/Relu                     3699.4   473.8  58.769
5    block2_pool/MaxPool                      1.6   923.5   3.316
6    block3_conv1/Relu                     1849.7   174.3  42.795
7    block3_conv2/Relu                     3699.4   179.7  82.962
8    block3_conv3/Relu                     3699.4   180.9  82.437
9    block3_pool/MaxPool                      0.8   931.8   1.644
10   block4_conv1/Relu                     1849.7   136.1  41.907
11   block4_conv2/Relu                     3699.4   170.0  67.074
12   block4_conv3/Relu                     3699.4   170.2  66.985
13   block4_pool/MaxPool                      0.4   907.0   0.845
14   block5_conv1/Relu                      924.8   312.0  19.970
15   block5_conv2/Relu                      924.8   315.8  19.730
16   block5_conv3/Relu                      924.8   312.7  19.929
17   block5_pool/MaxPool                      0.1   894.4   0.215
18   fc1/Relu                               205.5  2158.5  90.830
19   fc2/Relu                                33.6  2137.5  14.978
20   predictions/BiasAdd                      8.2  2642.3   2.960
21   predictions/Softmax                      0.0    19.0   0.201
-----------------------------------------------------------------
                     Total inference time                  749.64
-----------------------------------------------------------------
Generating Profile Report 'output_report.html'...
```

Do prediction on an image:

```
$ python predict-vgg16.py ~/Downloads/th.jpeg
```

## Reference

+ [ardamavi/Intel-Movidius-NCS-Keras](https://github.com/ardamavi/Intel-Movidius-NCS-Keras)
