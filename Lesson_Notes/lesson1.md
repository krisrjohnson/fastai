lesson1.ipynb

python 3.6 f

python code to train cats v dogs model: 
```python
		arch=resnet34
		data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
		learn = ConvLearner.pretrained(arch, data, precompute=True)
		learn.fit(0.01, 3)
```

3 epochs: looks at entire set of images 3 times
	prints accuracy on validation set
	1st value is value of loss function

fastai library runs on top of pytorch, written by fb

analyzing results

		`data.classes` #shows which label corresponds to 0 and to 1 (cats, dogs in this case)

models return log of predictions rather than probability itself
to get probability e^
`np.exp(learn.predict()[:,1])` numpy exponential of first prediction

first thing after creating a model is to visualize what the model's built

Go through all 7 videos asap
Do as much coding as possible
vision learning is more flexible than you'd think
Deep learning != machine learning, deep learning is a kind of machine learning
underlying function of deep learning is a neural network
	sigmoid fn = 1/(1-e^x)
		activation fn
Gradient descent of loss function to find the minimum (local and global minimums exist)
	for NN in particular, there aren't multiple diff local minima!

GPUs 10x faster than CPUs (GTX-1080i ~ $700, CPU that's 10x slower is $4k!)

Deep learning basic utility:
	fraud detection, sales forecasting, product failure prediction, pricing, credit risk, customer retention/churn, recommendation systems, ad optimization, anti-money laundering, resume screening, sales prioritization, call center routing, store layout, store location optimization, staff scheduling

key piece of a CNN is the convolution
	http://setosa.io/ev/image-kernels
	http://neuralnetworksanddeeplearning.com/chap4.html 

	nowadays use RELU instead of sigmoid
		y=max(x,0) #replace negatives w/ 0
		RELUctified Linear Unit
		As you change values of linear fns, allows you to build arbtirarily tall/thin blocks 
			and then combine them together, to create a universal approximation
			key idea behind why NNs can solve any computible problem

	Gradient descent - take derivative at a point to find direction to take a small step
		l = learning rate
		
		vid: https://www.youtube.com/watch?v=IPBSB1HLNLo
		107:24

with open("text.txt", "w") as textfile:
  textfile.write("Success!")
