# food_classifier

## web-app
Have started a classifier already to identify vegetables (started w/ cucumbers vs pickles :lol:)

Trying to create a Heroku webapp to be able to check an img from anywhere!

#### Deploying on Heroku:

On Windows and Mac:
In the Docker Quickstart Terminal - in the root folder of the web-app, build the web-app and push locally:
```
docker build -t app .
docker run -it app -p 5000:5000
```

To deploy on heroku with [Heroku CLI](https://en.wikipedia.org/wiki/Command-line_interface "cmd.exe or terminal") installed - login, create the web-app if it doesn't exist, and deploy.
```
heroku login
heroku container:login

APP_NAME="food-img-classifier"
heroku create $APP_NAME

heroku container:push web --app ${APP_NAME}

heroku container:release web --app ${APP_NAME}
heroku open --app $APP_NAME
heroku logs --tail --app ${APP_NAME}
```

### Foods Classified
TODO: Create a classifier on food groups to preprocess training? Aka, classifier with food pyramid categories: [meat, dairy, vegetables, grains, fruits]. Then use corresponding categories classifier. Would this allow higher classification accuracy per category?

Types of vegetables [link][digitalcommons.usu]
- Perennial (edible plant stem) Vegetables (asparagus, rhubarb, horseradish)
- Root Crops (carrots, betes, radish, rutabaga, turnip parsnip, sweet potato, and yam)
- Legumes (peas, beans, soybeans, limabeans)
- Bulb Crops/Alliums (onion, garlic, shallot, leek, chives, scallions)
- Salad Crops (lettuce, celery, swiss chard, parsley, endive, chicory, dandelion)
- Leafy Greens (spinach, kale, NZ spinach, collards)
- Cucurbits (squash, cucumbers, melons, cataloupe, pumpkins, winter squash, summer squash, zucchini)
- Cole/Cruciferous Crops (cabbage, cauliflower, broccoli, brussel sprouts, kale, kohlrabi, bok choy)
- Solanaceous (tomato, pepper, eggplant, tomatillo, potato)

Checklist: 
'asparagus','rhubarb', 'horseradish', 'carrot', 'beet', 'radish', 

#to go
'rutabaga',
'turnip','parsnip','sweet_potato','yam', 'pea','bean','soybean','limabean', 'onion','garlic','shallot',
'leek','chive','scallion', 'lettuce','celery','swiss_chard','parsley','endive','chicory','dandelion',
'spinach','kale','New_Zealand_spinach','collards', 'squash','melons','cantaloupe','pumpkins',
 'winter_squash','summer_squash','zucchini', 'cabbage','cauliflower','broccoli','brussel_sprouts',
'kale','kohlrabi','bok_choy', 'tomato','pepper','eggplant','tomatillo','potato'




Leafy green – lettuce, spinach and silverbeet.
Cruciferous – cabbage, cauliflower, Brussels sprouts and broccoli.
Marrow – pumpkin, cucumber and zucchini.
Root – potato, sweet potato and yam.
Edible plant stem – celery and asparagus.
Allium – onion, garlic and shallot.



### citation
Heavily used for web deployment:
>Pattaniyil, Nidhin and Shaikh, Reshama, [Deploying Deep Learning Models On Web And Mobile](https://reshamas.github.io/deploying-deep-learning-models-on-web-and-mobile/), 2019

Unfortunately we chose to work on the same task of food classification :(  They've also done a bird classifier, which is another one I wanted to do!! Get out of my head strangers!



[digitalcommons.usu]: https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2374&context=extension_histall