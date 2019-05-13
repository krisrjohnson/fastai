# food_classifier

## web-app
Have started a classifier already to identify vegetables (started w/ cucumbers vs pickles :lol:)

Trying to create a Heroku webapp to be able to check an img from anywhere!

#### Deploying on Heroku:

On Windows and Mac:
In the Docker Quickstart Terminal and with Heroku [CLI](https://en.wikipedia.org/wiki/Command-line_interface "cmd.exe or terminal") installed in the root folder of the web-app, build the web-app
```
docker build -t app .
docker run -it app -p 5000:5000
```

Then to deploy on heroku, first login, then create the web-app if it doesn't exist. And lastly deploy.
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



### citation
Heavily used for web deployment:
>Pattaniyil, Nidhin and Shaikh, Reshama, [Deploying Deep Learning Models On Web And Mobile](https://reshamas.github.io/deploying-deep-learning-models-on-web-and-mobile/), 2019

Unfortunately we chose to work on the same task of food classification :(  They've also done a bird classifier, which is another one I wanted to do!! Get out of my head strangers!

