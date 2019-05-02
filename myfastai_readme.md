# My Deep Learning/Fast.ai Notes

<p>I'm taking Jeremy Howard and Rachel *insert name* fast.ai Practical Deep Learning for Programmers course, v3. The advanced course is out in June I believe. :heart:
</p>

### so far
<p>Importing data and getting ImageDataBunch's to cooperate has been a massive pain. After watching [video one][1] and [video 2][2] I tried to spin up a super quick classifier on images of baseball vs cricket. I initially scraped from google following this [post][3] by user lindyrock. I'm pretty sure I pip installed google_images_download package, but it may have already been there. Same story for chromedriver pkg. Data was stored simulatneously under the course-v3/nbs/dl1/datasets and /storage. But trying to use imageDataBunch.from_folder() was a nightmare I never got working. 
</p>

<p>After many days of frustration, including the frustration of waiting for paperspace to relaunch my gradient notebook, paperspace wiped the copies of the fastai notebooks I was working on, forcing me to restart. Perusing the forums trying to understand how to interact with paperspace's /storage, a post relayed that Jeremy had created a notebook from lesson 2 to upload img data and use imageDataBunch.from_folder(). Going through that notebook, I was able to train!
</p>

<p>The first video uses transfer learning from a model built on imageNet, hence normalizing our imageDataBunch's with imagenet_stats. Normalizing with imagenet_stats aligns the rgb/pixel values of our data to the imagenet data, without which could cause nullify the transfer learning.
</p>

### I can do deep learning!

### Possible projects
*Pick one! Do it really well! Make it fantastic!*
- What's in my fridge
	- image classifier for fridge/groceries
		- cucumbers vs pickles
		- types of squash
- alcohol behind the bar
	- image classifier for what booze bottles are on display (and therefore being sold) at a bar
	- OR text extractor from an image classifier for same purpose
- tree classifier
- animal tracker!
	- scat and pawprints!

# FoodID
<p>For say vegetables, how can I pull out/get the comp to identify certain patterns, such as how many points the leafs might have (for trees or poison ivy for instance)? Could that be fed in or extracted? Not made into a hard rule per se, but flagged as a tuple of confidence if we know there are predetermining identifiers?</p>

Once we get all the way to Id'ing every veggie in the pristine conditions of the grocery aisle, how do we go all the way to say knowing "dirty" veggies right in the fields? *Young vs old veggies?*


[1]: https://course.fast.ai/videos/?lesson=1
[2]: https://course.fast.ai/videos/?lesson=2
[3]: https://forums.fast.ai/t/tips-for-building-large-image-datasets/26688