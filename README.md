# Self-Attention-GAN-Tensorflow
Simple Tensorflow implementation of ["Self-Attention Generative Adversarial Networks" (SAGAN)](https://arxiv.org/pdf/1805.08318.pdf)


## Requirements
* Tensorflow 1.8
* Python 3.6

## Summary
### Framework
![framework](./assests/framework.PNG)

### Code
```python
  f = conv(x, ch // 8, kernel=1, stride=1, scope='f_conv')
  g = conv(x, ch // 8, kernel=1, stride=1, scope='g_conv')
  h = conv(x, ch, kernel=1, stride=1, scope='h_conv')

  s = tf.matmul(g, f, transpose_b=True)
  attention_shape = s.shape
  s = tf.reshape(s, shape=[attention_shape[0], -1, attention_shape[-1]])  # [bs, N, C]

  beta = tf.nn.softmax(s, axis=1)  # attention map
  beta = tf.reshape(beta, shape=attention_shape)
  o = tf.matmul(beta, h)

  gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

  x = gamma * o + x
```
## Usage
### dataset

```python
> python download.py celebA
```

* `mnist` and `cifar10` are used inside keras
* For `your dataset`, put images like this:

```
├── dataset
   └── YOUR_DATASET_NAME
       ├── xxx.jpg (name, format doesn't matter)
       ├── yyy.png
       └── ...
```

### train
* python main.py --phase train --dataset celebA --gan_type hinge

### test
* python main.py --phase test --dataset celebA --gan_type hinge

## Results
### ImageNet
<div align="">
   <img src="./assests/result_.png" width="420">
</div>

### CelebA (100K iteration)
![celebA](./assests/celebA.png)

## Author
Junho Kim
