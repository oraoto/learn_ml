#%%
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import time
import networkx as nx
import graphviz as gv

# from nnabla.contrib.context import extension_context
# cuda_device_id = 0
# ctx = extension_context('cuda.cudnn', device_id=cuda_device_id)
# nn.set_default_context(ctx)

#%% Config
z_dim = 100
epochs = 2
lr = 0.001
batch_size = 64

#%% Data
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.astype('float32').reshape(train_x.shape[0], 1, 28, 28) / 255.
idx = np.random.permutation(len(train_x))
train_x = train_x[idx]

def mnist_next_batch():
    idx = np.random.randint(0, len(train_x) // batch_size - 1)
    true_images = train_x[idx * batch_size:(idx + 1) * batch_size]
    return true_images

#%% Discriminator and Generator
def D(images, is_train=True):
    with nn.parameter_scope('D'):
        with nn.parameter_scope('conv1'):
            conv1 = PF.convolution(images, 32, (3, 3), pad=(3, 3), stride=(2, 2), with_bias=False)
            conv1 = PF.batch_normalization(conv1, batch_stat=is_train)
            conv1 = F.leaky_relu(conv1)
        with nn.parameter_scope('conv2'):
            conv2 = PF.convolution(conv1,  64, (3, 3), pad=(1, 1), stride=(2, 2), with_bias=False)
            conv2 = PF.batch_normalization(conv2, batch_stat=is_train)
            conv2 = F.leaky_relu(conv2)
        with nn.parameter_scope('conv3'):
            conv3 = PF.convolution(conv2, 128, (3, 3), pad=(1, 1), stride=(2,2), with_bias=False)
            conv3 = PF.batch_normalization(conv3, batch_stat=is_train)
            conv3 = F.leaky_relu(conv3)
        with nn.parameter_scope('conv4'):
            conv4 = PF.convolution(conv3, 256, (3, 3), pad=(1, 1), stride=(1,1), with_bias=False)
            conv4 = PF.batch_normalization(conv4, batch_stat=is_train)
        with nn.parameter_scope('output'):
            output = PF.affine(conv4, 1)
            output = F.sigmoid(output)
    return output

def G(z, is_train=True):
    z = F.reshape(z, [batch_size, z_dim, 1, 1])
    with nn.parameter_scope('G'):
        with nn.parameter_scope('deconv1'):
            dc1 = PF.deconvolution(z, 256, (4, 4), with_bias=False)
            dc1 = PF.batch_normalization(dc1, batch_stat=is_train)
            dc1 = F.leaky_relu(dc1)
        with nn.parameter_scope('deconv2'):
            dc2 = PF.deconvolution(dc1, 128, (4, 4), pad=(1, 1), stride=(2, 2), with_bias=False)
            dc2 = PF.batch_normalization(dc2, batch_stat=is_train)
            dc2 = F.leaky_relu(dc2)
        with nn.parameter_scope('deconv3'):
            dc3 = PF.deconvolution(dc2, 64, (4, 4), pad=(1, 1), stride=(2, 2), with_bias=False)
            dc3 = PF.batch_normalization(dc3, batch_stat=is_train)
            dc3 = F.leaky_relu(dc3)
        with nn.parameter_scope('deconv4'):
            dc4 = PF.deconvolution(dc3, 32, (4, 4), pad=(3, 3), stride=(2,2), with_bias=False)
            dc4 = PF.batch_normalization(dc4, batch_stat=is_train)
            dc4 = F.leaky_relu(dc4)
        with nn.parameter_scope('output'):
            output = PF.convolution(dc4, 1, (3, 3), pad=(1,1))
            output = F.sigmoid(output)
    return output

nn.clear_parameters()

#%%
G_solver = S.Adam(0.0002, beta2=0.5)
D_solver = S.Adam(0.0002, beta2=0.5)

G(nn.Variable([batch_size, z_dim]))
with nn.parameter_scope('G'):
    G_solver.set_parameters(nn.get_parameters())

D(nn.Variable([batch_size, 1, 28, 28]))
with nn.parameter_scope('D'):
    D_solver.set_parameters(nn.get_parameters())

ones = nn.Variable.from_numpy_array(np.ones((batch_size, 1)))
zeros = nn.Variable.from_numpy_array(np.zeros((batch_size, 1)))

#%%
def show_image(z=None):
    if not z:
        z = nn.Variable.from_numpy_array(np.random.randn(batch_size, z_dim))
    fake_images = G(z, False)
    fake_images.forward()
    p = D(fake_images)
    p.forward()
    plt.show()
    r = np.sum((p.d > 0.5).astype('int8').reshape((batch_size,))) / batch_size

    fig = plt.figure(figsize=(5, 5))
    fig.suptitle(str(r))
    gs = plt.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    plt.title(str(r))
    for i, image in enumerate(fake_images.d):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(image.reshape(28, 28), cmap='Greys_r')
    plt.show()

#show_image()

#%%
for i in range(10000):
    ## Fake image
    z = nn.Variable.from_numpy_array(np.random.randn(batch_size, z_dim))
    fake_images = G(z)
    predict = D(fake_images)
    fake_loss = F.mean(F.binary_cross_entropy(predict, zeros))

    D_solver.zero_grad()
    fake_loss.forward()
    fake_loss.backward(clear_buffer=True)
    D_solver.update()

    ## Real image
    true_images = nn.Variable.from_numpy_array(mnist_next_batch())
    predict = D(true_images)
    real_loss = F.mean(F.binary_cross_entropy(predict, ones))

    D_solver.zero_grad()
    real_loss.forward()
    real_loss.backward(clear_buffer=True)
    D_solver.update()

    # # G
    z = nn.Variable.from_numpy_array(np.random.randn(batch_size, z_dim))
    fake_images = G(z)
    predict = D(fake_images)
    g_loss = F.mean(F.binary_cross_entropy(predict, ones))

    G_solver.zero_grad()
    g_loss.forward()
    g_loss.backward(clear_buffer=True)
    G_solver.update()

    if i % 100 == 0:
        show_image()
        print('GAN train loss', i, g_loss.d, real_loss.d, fake_loss.d)
    if i % 500 == 0:
        nn.save_parameters('params_iter' + str(i) + '.h5')

#%% Interpolation
def interp(a, b, step):
    result = []
    delta = (b - a) / (step - 1)

    for i in range(step):
        result.append(a + i * delta)
    return np.concatenate(result)


zs = np.random.randn(4, z_dim)
z1, z2, z3, z4 = zs[0], zs[1], zs[2], zs[3]

zs = np.concatenate([z1, z2, z3, z4])
show_image(nn.Variable.from_numpy_array(np.resize(zs, (batch_size, z_dim))))

z = interp(interp(z1, z2, 8), interp(z3, z4, 8), 8)

show_image(nn.Variable.from_numpy_array(z))
