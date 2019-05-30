
possible ideas:
* bald vs non
	* can do old v young of same man to see possible visual cues for future baldness
* 

The first thing we do when we approach a problem is to take a look at the data. We always need to understand very well what the problem is and what the data looks like before we can figure out how to solve it. Taking a look at the data means understanding how the data directories are structured, what the labels are and what some sample images look like.

The main difference between the handling of image classification datasets is the way labels are stored. In this particular dataset, labels are stored in the filenames themselves. We will need to extract them to be able to classify the images into the correct categories. Fortunately, the fastai library has a handy function made exactly for this, ImageDataBunch.from_name_re gets the labels from the filenames using a regular expression.

*not sure where numpy got loaded in in this notebook, maybe fastai.vision/metrics?*

fastai.vision,nlp,tabulardata, 1 other
from fastai import * 	#since we're in a Jupyter notebook and want to exp, import everything (for prod, import only what you need)

data from academic datasets or kaggle datasets
	they have strong baselines for you to compare

fine grain classification - picking b/t mult categories
untar_data() 	#fn to import data
help(untar_data) 	#from fastai.datasets module

# Modeling
python3 has path objects which are easier to use than strings, ex:  path_anno = path/'annotations'
Everything you model will be a DataBunch object
	contains training data, test data, and optionally validation data
	normalize to get about the same mean and std deviation - helps w/ apply the fn equally to rgb channels
	if img size is not 224, ds_tfms transforms to same size
224x224 is an extremely common img size
models designed so final layer is 7x7!!
data bunch will always have a property call 'c'
	so data.c will be the # of classes
models are trained using a learner - for vision it's ConvLearner
	learn = ConvLearner(data, models.resnet34, metrics=error_rate) 	#says print out error rate
		on first run, downloading resnet34 *pretrained* weights
			trained looing at 1.5 mn pictures from imageNet
			this does __transfer__ learning
			using pretrained models which we use to refine further is *at least* 1/100th as fast!
	how do we know if we've overfit?
		using validation set - model never gets to look at while training
	resnet works extremely well almost all of the time
		__RESNET?!?__
	there are two, resnet34 and resnet50, choose the small one first to test

Always use two stage process: run a first pass using the pretrained model, where the og weights are frozen
Save off in between! 
Second stage: learn.unfreeze() 	#unfreezes the pretrained weights so now you're modeling all layers

# learn.fit_one_cycle(4) 	
* 2018 modeling!
* 4 is how many times does the model learn from the entire dataset, 
* can lead to overfitting since similar to epochs, if see same image too many times then will lead to overfitting

# Model Results
interp = ClassificationInterpretation.from_learner(learn) 	#learn object knows data and model(now trained)

[fastai jupyter notebooks](https://www.github.com/fastai/fastai_docs)

_easy way to find/pull images?_
	__Run the code!!!__
### wired AI fast.ai
### Allen Institute for AI
### introtodeeplearning.com 
### stanford dawnbench
### visualizaing and understanding convolutional networks - Matt Zeiler, CEO Clarify, NYU


# This week - 4/21/19
* run this notebook -x
* get own img dataset -x 
* Francisco guide dwnld data from goog imgs - x
