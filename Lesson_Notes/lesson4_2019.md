# #[Video 4][vid4]
___


### NLP -__REWATCH!!!!__
___
```python
%reload_ext autoreload
%
%

from fast.ai import
from fast.ai import text
```

Language model - model that learns to predict next word of a sentence.


### Tabular Data
___
"Feature engineering doesn't go away, but it becomes much simpler." Makes engineering a lot easier, hand created features aren't as necessary (still need some!). Makes things easier.

"Used to use Random Forests 99% of the time when using tabular data, now use Machine Learning 99% of the time." Nobodies made tabular data available in a library. 

Assume data is in a pandas data frame. Pandas can read from Spark, Hadoop, etc. Most common:
```python
df = pd.read_csv(path/'data.csv')
train_df, valid_df = df[:-2000].copy(), df[-2000:].copy() #aka, validation is last 2k rows
```

Q: 10% of cases where you would not default to NNs
A: I still tend to give NNs a try, but I dunno. May as well try both, NN and RF.

Tabular data example preprocessing. 
```python
path=untar_data(URLs.ADULT_SAMPLE) #in sample data provided by Fast.ai
df = pd.read_csv(path/'adult.csv')
 
dep_var = '>=50k' #salary > 50k
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'] # categorical var's
cont_names = ['age', 'fnlwgt', 'education-num'] #continuous variables
procs = [FillMissing, Categorify, Normalize]

test = TabularList.from_df(df.iloc[800:1000].copy(), path=path, cat_names=cat_names, cont_names=cont_names)
data = TabularList.from_df(df, path=path, cat_names = cat_names, cont_names=cont_names, procs=procs #this is a databunch
	.split_by_idx(list(range(800,1000)))
	.label_from_df(cols=dep_var)
	.add_test(test, label=0)
	.databunch())
data.show_batch(rows=10) #show 10! displays output obvi
```
Transforms in computer vision flip, brighten, etc (convolutions). For tabulr data the parallel is processes, aka preprocessing the data frame, df, rather than doing as we go. Do once ahead of time.

FillMissing - looks for missing values and deals (replace w/ median and a binary col of whether missing or not)
Categorify - turns categorical variables into pandas categories
Normalize - subtract mean and divide by std_dev

_get_tabular_learner_
```python
learn = get_tabular_learner(data, layers=[200,100], metrics=accuracy)
learn.fit(1, 1e-2) #1 epoch, lr =.01
# 3 secs, acc 82%
```

Q: How to combine NLP tokenized w/ meta data like tabular data, i.e. imdb (actor, year made, genre, etc).
A: not there yet, conceptually, same as combining categorical and continuous variables, two diff sets of inputs merging into some layer.

Q: scikit learn and xgboost outdated in future? 
A: no idea, not good w/ predictions (laughter). I'm not an ML model ;)

_fast.ai has no scikit learn dependencies_

##### Pseudo Code
___
```python
df = pd.read_csv('some/csv')
dep_var, cat_name, cont_names
procs
data = TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names)

```




# Collab Filtering
Basic version, have 2 cols (user_id, item_id). Item can be movie, product, etc. Can add review, time_code, stars, etc. 

Ex, userid, movieid, rating, timestamp (will ignore ts for now)
```python
path=untar_data(URLs.ML_SAMPLE)
ratings = pd.read_csv(path/'ratings.csv')
learn = get_collab_learner(ratings, n_factors=50, min_score=0., max_scores=5.)
learn.fit_one_cycle(4, 5e-3) #5s, 4 epochs, lr=.005
#can now pick a userid, movieid, and guess whether that user will like that movie
```

Cold start problem - Main problem with collab learning is predictions for either a new user or a new movie, where you have no data!

Only conceptual way to solve it is a 2nd model, a meta data model, for new users or new movies! Netflix used to ask you to rate 20 generic movies!

For new movies, first 100 go in and say they liked it, then the data starts to roll in. For products, you can use a metadata based tabular model based on demographics (geo, age, sex, etc). 

Collab is _specifically_ for once you have info on users and whatever you're trying to predict.

Q: Timeseries on tabular data, is there an RNN?
A: next week - short: don't use an RNN for timeseries tab, but extract columns for day_of_week, is_weekend, holiday, store_open, etc
  Adding extra cols gives state of the art results. RNNs for timeseries exist, but not for tabular style retail or logistics dbs.

Q: Source to learn more abou cold start?
A: Lookup - check forums


#### Break-1:07
Rest of course is gonna be digging deeper into key applications and theory/source code.



<!-- links -->

[vid4]: https://course.fast.ai/videos/?lesson=4 'fast.ai'