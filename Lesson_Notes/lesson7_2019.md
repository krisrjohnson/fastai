# [Video 7][vid7]
## Generator!!!!!
Create one for ma face!

## GANs!!!!!!
1hr mark

GANs train two models, a binary cross entropy classifier called a critic who returns a 1 or 0 for real or fake image!

Then we can update the generator based on whether the images it's producing are real or fake!

When you're doing a GAN need to be particularly careful teh generator and critic can't push the weights in the same way. Have to use spectral normalization to make gans work nowadays. gan_critic(), fastai will use a binary classifier for you. Anytime you're doing a GAN have to wrap your critic loss with adaptive loss `loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())`

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



- [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
- [Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic) paper shown in class
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [Interview with Jeremy at Github](https://www.youtube.com/watch?v=v16uzPYho4g)
- [ipyexperiments](https://github.com/stas00/ipyexperiments/) - handy lib from @stas that is even better than `gc.collect` at reclaiming your GPU memory
- [Documentation improvements thread](https://forums.fast.ai/t/documentation-improvements/32550) (please help us make the docs better!)

[vid7]: 'https://course.fast.ai/videos/?lesson=7'