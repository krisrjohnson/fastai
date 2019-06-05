# [Video 7][vid7]
## Generator!!!!!
Create one for ma face! GANs!!!!!!

- restarting from beginning

### RESNETs From Scratch
_________________
Using what we've learned, create a model from scratch.

#### Preprocessing
```python
#img size is 28x28
il = ImageList.from_folder(path, convert_mode='L') #args go to Pillow, L is convert mode for B+W
defaults.cmap='binary' #default color map for fast.ai, refer to matplotlib
il.items[0]; il #posix path; displays info; pytorch puts channel first
# What is channel? All the convs do rank3 tensors, so auto places a unit vector
il[0].show() #displays pic
sd = il.split_by_folder(train='training', valid='testing') #kaggle nomenclature. valid is for testing your model, test set is in the wild. Test has no labels
ll = sd.label_from_folder() #folder for each class w/in 'training' dir
```

```python
x,y = ll.train[0] #y is cat, x is img object
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
'''for small imgs, rand_padding, which returns two transforms bit that does padding \
 and bit that does random crop so use asterisk to say put both these transforms in this \
  list. Empty bracket is no tfms for validation Style'''

ll = ll.transform(tfms)
bs = 128
data = ll.databunch(bs=bs).normalize() #not using pretrained so no imagenet_stats normalization. This grabs a batch and creates a mean and std_dev to normalize w/
x,y = data.train_ds[0]

'''each time you grab from training set, loads from disk and tfms on the fly, hence the result is of plot_multi is 9 slightly diff figures'''
def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')
plot_multi(_plot, 3, 3, figsize=(8,8))

xb,yb = data.one_batch() #bs is 128 ^
data.show_batch(rows=3, figsiz=(5,5)) #all datablock apis have show_batch()
```

#### training
In the past we've used resnet directly plus a few layers, instead of trainig a full new model.

Going to standardize the convolution we'll use, in this case kernel_size=3, stride=2, and padding=1. Stride=2 means we're skipping over a pixel to perform the convolutions, so our grid size will be halved on each conv layer.

The input is 1 channel since we start with a single gray scale image. We can pick however many filters we want, in this case we're starting with 8
```python
def conv(ni, nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1) #going to use multiple convs, so define/std ahead of time
model = nn.Sequential(
	conv(1, 8), #8x14x14 - started w/ a 28x28 and conv's halve it, so 14x14
	nn.BatchNorm2d(8),
	nn.ReLU(),
	conv(8, 16), #16x7x7
	nn.BatchNorm2d(16),
	nn.ReLU(),
	conv(16, 32), #32x4x4 #b/c padding =1
	nn.BatchNorm2d(32),
	nn.ReLU(),
	conv(32, 16), #16x2x2
	nn.BatchNorm(16),
	nn.ReLU(),
	conv(16, 10), #10x1x1 #Predicting digits 0-9, or 10 numbers!
	nn.BatchNorm2d(10),
	Flatten() 	#remove (1,1) grid
	)

learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
learn.summary() #displays layers, w/ shape
```
nn.BatchNorm2d()?

```python
xb = xb.cuda() #xb is a minibatch we did earlier
model(xb).shape # torch.Size([128,0])
learn.lr_find(end_lr=100)
learn.recorder.plot() #pick lR

learn.fit_one_cycle(3, max_lr=0.1) #3 epochs, .1 is steepest part
```

#### Refactor
Instead of conv, batchnorm, relu repeat, fast ai has conv_layer()
```python
def  conv2(ni, nf): return conv_layer(ni, nf, stride=2)
model = nn.Sequential(
	conv2(1, 8),
	conv2(8, 16),
	conv2(16, 32),
	conv2(32, 16),
	conv2(16, 10),
	Flatten()
	)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
learn.fit_one_cycle(10, max_lr=0.1)
```

How can we improve this? Deeper network. After every stride 2, do a stride 1 since it doesn't change the feature map as you're using all pixels. Basically add in an extra layer of conv filters.

Problem pointed out in [this][deepreslearning] paper. More params should overfit, but it is worse than the shallow network. The authors tested a skip connection that has to be as good as the shallow network. This is what ResNET is! This skip connection is a ResBlock!

[Visualizing loss][losspaper] paper, shows how skip connections massively smooth out the loss. The overfitting network w/out skip connections gets stuck in hills and valleys! Took 3 years after the fact why skip connections work!

Keep refactoring architectures as you go!!!!! Makes life easier

ResBlock with refactoring:
```python
def ResBlock(nn.Module):
	def __init__(self, nf):
		super().__init__()
		self.conv1 = conv_layer(nf, nf)
		self.conv2 = conv_layer(nf, nf)

	def forward(self, x): return x + self.conv2(self.conv1(x))

# help(res_block) #fastai already has fn
def conv_andres(ni, nf): return nn.Sequential(conv2(ni,nf), res_block(nf))
model = nn.Sequential(
	conv_and_res(1, 8),
	conv_and_res(8, 16),
	conv_and_res(16, 32),
	conv_and_res(32, 16),
	conv_and_res(16, 10),
	Flatten()
	)

learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
learn.lr_find(end_lr=100)
learn.recorder.plot()
learn.fit_one_cycle(10, max_lr=0.1)
```

DenseBlock:
fastai source code has a MergeLayer in the fwd mode. SequentialEx is Sequentiale Extended, that stores input in input.orig. Dense will concatenate instead of addition, creating a dense block and DenseNet (instead of ResNet). So effectively multiplies datasize by however many denseblocks used. Memory intensive but have very few parameters, work really well on small datasets. Really well for segmentation, where you want to reconstruct original resolution of your picture, has original pixels!!


#### Summary of this section - ResNET defined
Using the MNIST digits dataset, a very simple and std dataset of 28x28 black and white images of hand drawn digits 0-9. Attempting to create a model that will accurately predict hand drawn digits. Normalizing the data with std deviation and mean drawn from a random sample.

Using standard convolutions to encode the data down from 28x28 to a 10x1 vector who's index with the max value is th models prediction. Going from 1 channel (a single image) all the way to 32 filters and back down to 10. Regular convolution, batch normalization, rectified linear units for our layers. Cross Entropy for our loss function.

Using Residual blocks we can achieve even better results since they smooth the loss! Otherwise there are a lot of local peaks and valleys! Residual blocks perform two CNNs in the same parameter space as the input and then add back the original input.

Dense blocks are even more computationally heavy as they expand the data set with every additional layer.

__Stopping at 33:31__


### U-Nets - Convolutions for Segmentation

Original U-Net downsampling path does 5 steps, doubling the channels at each step starting from 64 and halving the resolution.

To upsample, we do a deconvolution/half-stride conv/transpose convolution. The paper, [Guide to conv arithmetic][conv_ar], shows what a stride half conv looks like. This is how people used to do upsampling. Nowadays, nearest neighbor interpolation is used for upsampling, which involves say quadrupling the resolution (32x32->64x64) by copying each original pixel into a square. Then a stride 1 convolution can be trained. Stride1 means moving over a 2x2.

U-Nets use cross connections (similar to dense blocks) for every downsampling, meaning the upsampling has the original pixel from the downsample point. So the last upsample has the original pixels! (Well after the first couple of 3x3 convolutions).

Generally speaking we want as much interaction as possible "50:14". Stride2 conv, can't keep dense netting. In practice a dense block doesn't keep all the info all the way through, but up until stride2 convs.

### GANs and RNNs

```python
def crappify(fn, i):
  dest = path_lr/fn.relative_to(path_hr)
  dest.parent.mkdir(parents=True, exist_ok=True)
  img = PIL.Image.open(fn)
  targ_sz = resize_to(img, 96, use_min=True)
  img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
  w,h = img.size
  q =random.randint(10,70)
```

`parallel(fn, il.items)` is a fastai method. Runs the fn on the list of things in parallel!
Using `MSELossFlat()` which turns img and channels into a long vector so we can do MSE pixel by pixel. Using U-Net and pretrained resnet-34. This MSE loss fn doesn't do what we want, since most of the pixels are very nearly the right color, only the white numbers will be far off.

So we'll use a GAN to call a critic to say how real or fake an image looks. Can we fool the critic w/ our generated images! Pretrained both the Generator and the Critic! Pain to train is at the start, because it's the blind leading the blind!

GANs train two models, a binary cross entropy classifier called a critic who returns a 1 or 0 for real or fake image!

Then we can update the generator based on whether the images it's producing are real or fake!

When you're doing a GAN need to be particularly careful the generator and critic can't push the weights in the same way. Have to use __spectral normalization__ to make gans work nowadays. gan_critic(), fastai will use a binary classifier for you. Anytime you're doing a GAN have to wrap your critic loss with adaptive loss `loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())`

```python
def get_crit_data(classes, bs, size):
    src = ImageList.from_folder(path, include=classes).random_split_by_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=size)
           .databunch(bs=bs).normalize(imagenet_stats))
    data.c = 3
    return data
data_crit = get_crit_data([name_gen, 'images'], bs=bs, size=size)

loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())
def create_critic_learner(data, metrics):
	return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_critic, wd=wd) #gan_critic from fast.ai lib
learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand) #accuracy metric for gans
learn_critic.fit_one_cycle(6, 1e-3)
```

`gan_learner()`, pass in generator and critic and it'll figure out how much to train gan then critic. Add together pixel loss and critic loss, so scale the pixel loss (mutliplier between 50 and 200). GANs hate momentum. Use the hyperparams from lesson7-superres-ganipynb.

GAN loss numbers are _meaningless_! Numbers should stay about the same, since the two models are fighting each other.

Obvious problems:
Eyeballs. Critic doesn't know about eyeballs.

Q: when not to use U-Nets
A: U-Net - size of output is similar to size of input. Any generative modeling (segmentation).

_lesson7-wgan.ipynb - implementation of WGAN w/ fastai library. This generates images from nothing. Approach is can we create a bedroom. Can I create my face!!!! Input is random noise._

Loss fn that does a good job of saying this is a high quality image and looks like the thing it's meant to. Perceptual Losses for Real-Time Style Transfer and Super-Resolution (in fastai, referred to as feature losses). After we go through our generator (U-Net-ish thing, also called an encoder-decoder), we take prediction and put it through pretrained imageNet network (VGG for this paper, old but still used). Output of that is a classification.

_feature loss_: Goes through lots of layers, but takes the activations of somewhere in the middle. Then we take the target (the actual y value), put it through same feature map, and do MSE vs generated image. So it basically does a feature comparison (aka feature loss). Walkthrough in l7-superres.ipynb

#### lesson7-superres.ipynb

`vgg_m = vgg16_bn(True).features.cuda().eval()` - take vgg features, throw it on the gpu w/ `.cuda()` and in `.eval()` mode since we're not training. Want to grab features from every time just before grid size changes.

So now we create a `FeatureLoss` class where we'll pass it some pretrained model `m_feat`, the model which contains the features which we want our feature loss on. We grab intermediate layers in python by hooking them.

Train, unfreeze and train some more, double res and halve batch size, train again, unfreeze, etc. This beats a GAN and quicker!

Can now upsize a medium res (instead of low res)! Adds texture and does it! Hour and half of training. This leads to DeOldify, from a 2017 student, which they've helped him w/ pretraining etc so should be even better!



#### RNN
Important to be able to calculate shape of each layer. When two sets of activations come together, you can either add or concatenate. Lesson7-human-numbers.ipynb, all the numbers 1-9999 written out in english. Model will predict next word, a toy example.

bptt -> backprop three times (aka using 3 words to predict 4th). So we take batch size and then bptt to create the actual mini-batch size, which is total training data/bs/bptt. DataLoader slightly randomizes bptt. So the y1 is one offset from x1, since we're trying to predict next word. Every mini-batch joins up with the previous mini-batch.

Original RNN, take first word (w1) and put it through an embedding. Take second word (w2) and put it through an embedding and add it to w1's embedding. Put that combined through an embedding. Take w3, embed, add to output of previous step, and embed that. Result is predicted word.

Can abstract and do a loop of those take word, embed, add to previous answer. Can further abstract by taking the prediction from each embedding and appending to an array, keeping the hidden layer output (aka move h into the constructor).

Abstract even further! Can RNN into an RNN! Deep networks have awful bumpy loss layers, so add skip connections to smooth loss. Can add a GRU (LSTM) to decide how much of the previous word to keep, etc.

RNNs are good for sequence labeling tasks. Skip derivations theorems and lemmas! Write english prose for your understanding from 6 months ago. Get together with others.



- [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
- [Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic) paper shown in class
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [Interview with Jeremy at Github](https://www.youtube.com/watch?v=v16uzPYho4g)
- [ipyexperiments](https://github.com/stas00/ipyexperiments/) - handy lib from @stas that is even better than `gc.collect` at reclaiming your GPU memory
- [Documentation improvements thread](https://forums.fast.ai/t/documentation-improvements/32550) (please help us make the docs better!)

[vid7]: https://course.fast.ai/videos/?lesson=7
[deepreslearning]: https://arxiv.org/abs/1512.03385
[losspaper]: https://papers.nips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets.pdf 'visualizing loss'
[conv_ar]: https://arxiv.org/pdf/1603.07285.pdf 'dumb way to do it'
