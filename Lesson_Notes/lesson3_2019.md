#[Video 3][vid3]

Today (5/30/19):
- vid3
- vid4
- kaggle comp for removing bkg
- fast.ai GAN using my face (update of colab)
- audio of fast.ai on dropbox, and downloaded to phone.
EC: pdf -> text and uploaded to personal website (for ease of reading pdfs on the go)

##datablock.ipynb Notes
___
torch class: DataLoader
torch.utils.data.DataLoader  requires dataset (dataobject), batch_size

fast.ai class: DataBunch - binds a train_dl and valid_dl (both DataLoader objs)

If you have images in train and valid parent dirs, ex pseudo-code (need to be in a certain order):
```python
data = (ImageFileList.from_folder(path)
	.label_from_folder()
	.split_by_folder()
	.add_test_folder()
	.datasets()
	.transform(tfms, size=224)
	.random_split_by_pct()
	.databunch(bs=64))

data.train_ds[0] #data in the loader
data.batch() #display 
data.valid_ds.classes
data.show_batch
```
`Shift+tab` shows details of fn in ipynb, `??` will give source code
Documentation URL format is docs.fast.ai/NAMEOFCLASS.html. So fro get_transforms(), [docs.fast.ai/get_transforms.html](docs.fast.ai/get_transforms.html 'fast.ai get_transforms() doc')

tfms = get_transforms 

metrics do not affect the model, only display how you're doing. Accuracy is most obvious.

For satellite imagery, can't use argmax() since we're not looking at one feature (like in classification). We're going to pick a threshold and for any of the 17 features we're looking at, if they pass the threshold then the model is saying the image has that feature.

data.c is how many outputs are model will create, in this case 17.

### Satellite Imagery Classification
___
`def acc_o2(inp,targ): return accuracy_thresh(inp,targ,thresh=0.2)` is so common that we can just create a partial of accuracy_thresh, `acc_02 = partial(accuracy_thresh, thresh=0.2)`. This will have the exact same functionality but calls it with thresh=0.2 kwarg. Very common in the fast.ai library.

`learn.save('model_name-stage1')`, `learn.unfreeze() #when using resnet`, `learn.fit_one_cycle(5, slice(1e-5, lr/5))`, is a common path for resnet, you're unfreezing the first pretrained layers. `learn.lr_find(); learn.recorder.plot()` after unfreeze, you want 10x before it shoots up for first half and the lr from the first model divided by 5. These are discriminative learning rates.

Q: resources for video, like pulling frames. A: Depends. If using the web, then web APIs. Client side, OpenCV.

<!-- Transfer learning: from 128->256 -->
Trained alot, verge of overfitting. Going up by rez of 2 is new dataset w/ 4x as much data. Keep the same learner, but chg the databunch to 256x256. By splitting data source (src) and DataBunch object, can create a new DataBunch object by just using the same src, but changing the tfms to size=256!

Using the same learner, updating the data in the learner (`learn.data = data)`, freezing the first few layers (how does it know? `learn.freeze()`), finding a new LR `learn.lr_find(); learn.recorder.plot()`, and running one cycle `learn.fit_one_cycle(5, slice(lr))`


### CAMVID Segmentation - min 57
___
Segmentation is a classification problem per pixel, as in the output per pixel is an id corresponding to a class we define. 

To build a segmentation model, someone has to go in and label for the raw data. This is a lot of work.

Data retrieval, reg imgs were some 11 digit alphanumeric format, label imgs (segmentation view) were same but w/ underscore `_P` so combined w/ a quick lambda fn.
```python
path_lbl = path/'labels'
path_img = path/'images'
fnames = get_image_files(path_img) #fast.ai method?
lbl_names = get_image_files(path_lbl)
get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'
mask = open_mask(get_y_fn(img_f)) #file contains ints, so open_mask()
mask.show(figsize=(5,5), alpha=1) #fast.ai can handle masks
```


If you add `.to_fp16()` to any Learner call, will do in 16 bit precision. Very new, need most recent CUDA drivers. A really recent GPU it'll work 2x as fast (RTX 2080 TI [$1200])


### BIWI Center of Face
___
Create a model to find center of face. This is not a classification model, but a regression. Regression means any model where your output is a number or a set of numbers. Whenever you have a predfict from image, you'd do a similar model. Just a resnet cnn.

Not sure how this is better/faster than dlib and Facial landmarking, but adds some sense of pose I guess?

They use kinect w/ depth sensor. They have some conversion routines, which are from the README of BIWI. 

`def get_ip(img,pts): return ImagePoints(FlowField(img.size, pts), scale=True)` returns X,Y coordinates

cross_entropy loss: did you predict the correct class and how confident were you?
MSE: since not classification, MSE is distance off (kinda)


Q: What's normalization should we use? Can we use pretrained models on different datasets? 
A: Pretrained models you can definitely use, and w/ imagenet_stats. Because the pretrained models were trained off imagenet! For any pretrained model, you want to normalize, aka mean of 0 and std_dev=1 for all three channels (RGB). __Always use the same stats the model was trained with.__

## Summary
____
All of these, gradient descent with minibatches (so SGD) to fit some parameters of the model. Those params are basically matrix multiplcations.
No amount of matrix mult is possible going to be able to read IMDB movie reviews and decide positive or negative, or say a road is in a satellite image. Every matrix multipication is a linear model. A linear on top of a linear is just linear. No amount of stacking matrix multiplications will help.

What're we actually doing? All we're doing is matrix multiplication, but after each one we do a non-linearity (the activation fn). Takes the result, and sticks it through a fn. In the old days they used sigmoids (tanh, sigmoid). Nowadays we use ReLUs, which are similar in shape to sigmoid, but actually transform it into a line (really y=max(x,0)). Doing this a bunch of times is a deep learning neural network. 

The _Universal Approximation Theorem_: stacks of linear fns and non-linearities can approximate any fn arbitrarily closely.

Because the weight matrices are millions of parameters, the intuition requires accepting the empirical nature of the approximation theorem. 

Link: [Activation FNs in NNs](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)



<!-- Links -->

[vid3]:https://course.fast.ai/videos/?lesson=3 'fast.ai'


