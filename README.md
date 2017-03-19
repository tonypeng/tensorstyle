# tensorstyle
Fast image style transfer using TensorFlow!

Both the "slow," iterative method described in the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et al. and the "fast" feed-forward version described in the paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) by Johnson et al. with modifications (e.g. replication border padding, instance normalization described in [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022) by Ulyanov et al, etc.) are implemented.

To run, download the [VGG-19 network](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) into `models/vgg` and the [COCO dataset](http://mscoco.org/), then edit the parameters in `train.py` (should be self-explanatory), then run `python3 train.py`. To run the iterative method, uncomment the relevant sections in the code as indicated by the comments.

Note: Minimal hyperparameter tuning was performed. You can probably get much better results by tuning the parameters (e.g. loss function coefficients, layers/layer weights, etc.).


### Differences from Paper
* ELU instead of ReLU (experimental)
* [Resize convolution layers](http://distill.pub/2016/deconv-checkerboard/) instead of transpose convolution layers (experimental)
* VGG-19 and content/style layers specified in *Gatys et al.*
* Instance normalization for all convolutional layers including those in residual blocks
* Output tanh activation is scaled by 150 then centered around 127.5 and clipped to [0, 255]
* Border replication padding (reflection padding is planned)
* Batch size 16 for 2 epochs (~10000 iterations)

### Samples

*White Line*, Fast, Content 5:Style 85:Denoise 5

<p>
<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/WhiteLine/style.jpg" height="221" style="padding-right: 8px">
<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/WhiteLine/WorldTradeCenter.jpg" width="332">
<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/WhiteLine/WorldTradeCenterStyled.jpg" width="332">
</p>


*Udnie*, Fast, Content 5:Style 85:Denoise 1

<p>
<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/Udnie/style.jpg" height="186" style="padding-right: 8px">
<img src="https://github.com/tonypeng/tensorstyle/blob/master/examples/Udnie/BostonSkyline.jpg" width="330">
<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/Udnie/BostonSkylineStyled_ELU_ResizeConv.jpg" width="330">
</p>

*Red Canna*, Fast, Content 5:Style 85:Denoise 5

<p>
<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/RedCanna/style.jpg" height="220" style="padding-right: 8px">
<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/RedCanna/EiffelTower.jpg" width="330">
<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/RedCanna/EiffelTowerStyled.jpg" width="330">
</p>

*Udnie*, Fast, Content 5:Style 50:Denoise 50 (ReLU, Transpose Convolution)

<p>
<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/Udnie/style.jpg" height="230">
<img src="https://github.com/tonypeng/tensorstyle/blob/master/examples/Udnie/GoldenGate.jpg" width="307">
<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/Udnie/GoldenGateStyled.jpg" width="307">
</p>

*A Sunday on La Grande Jatte*, Slow, Content 0.005:Style 1: Denoise 1

<p>
<img src="https://github.com/tonypeng/ml-playground/blob/master/style-transfer/ASundayOnLaGrandeJatte.jpg?raw=true" height="176">
<img src="https://github.com/tonypeng/ml-playground/blob/master/style-transfer/KillianCourt.jpg?raw=true" width="280">
<img src="https://raw.githubusercontent.com/tonypeng/ml-playground/master/style-transfer/KillianCourt%2BASundayOnLaGrandeJatte_Denoised.jpg" width="280">
</p>

Playing around with patterns:

<p>
<img src="https://s-media-cache-ak0.pinimg.com/736x/48/d8/2e/48d82e9b70c9762cafc305f6ecc7aff2.jpg" height="280">
<img src="https://graph.facebook.com/100001233078747/picture?type=large" width="280">
<img src="http://i.imgur.com/1M3Yed0.png" width="280">
</p>


### License
[MIT](https://github.com/tonypeng/tensorstyle/blob/master/LICENSE)
