# tensorstyle
Fast image style transfer using TensorFlow!

Both the "slow," iterative method described in the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et al. and the "fast" feed-forward version described in the paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) by Johnson et al. with modifications (e.g. replication border padding, instance normalization described in [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022) by Ulyanov et al, etc.) are implemented.

To run, download the [VGG-19 network](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) into `models/vgg` and the [COCO dataset](http://mscoco.org/), then edit the parameters in `train.py` (should be self-explanatory), then run `python3 train.py`. To run the iterative method, uncomment the relevant sections in the code as indicated by the comments.

Note: Minimal hyperparameter tuning was performed. You can probably get much better results by tuning the parameters (e.g. loss function coefficients, layers/layer weights, etc.).

Samples:

*Udnie*, Fast, Content 5:Style 75:Denoise 50

<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/Udnie/style.jpg" height="186">
<img src="https://github.com/tonypeng/tensorstyle/blob/master/examples/Udnie/BostonSkyline.jpg" width="330">
<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/Udnie/BostonSkylineStyled_75.jpg" width="330">

*Udnie*, Fast, Content 5:Style 50:Denoise 50

<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/Udnie/style.jpg" height="230">
<img src="https://github.com/tonypeng/tensorstyle/blob/master/examples/Udnie/GoldenGate.jpg" width="307">
<img src="https://raw.githubusercontent.com/tonypeng/tensorstyle/master/examples/Udnie/GoldenGateStyled.jpg" width="307">

*A Sunday on La Grande Jatte*, Slow, Content 0.005:Style 1: Denoise 1

<img src="https://github.com/tonypeng/ml-playground/blob/master/style-transfer/ASundayOnLaGrandeJatte.jpg?raw=true" width="280">
<img src="https://github.com/tonypeng/ml-playground/blob/master/style-transfer/KillianCourt.jpg?raw=true" width="280">
<img src="https://raw.githubusercontent.com/tonypeng/ml-playground/master/style-transfer/KillianCourt%2BASundayOnLaGrandeJatte_Denoised.jpg" width="280">


Playing around with patterns:


<img src="https://s-media-cache-ak0.pinimg.com/736x/48/d8/2e/48d82e9b70c9762cafc305f6ecc7aff2.jpg" height="280">
<img src="https://graph.facebook.com/100001233078747/picture?type=large" width="280">
<img src="http://i.imgur.com/1M3Yed0.png" width="280">



License: [MIT](https://github.com/tonypeng/tensorstyle/blob/master/LICENSE)
