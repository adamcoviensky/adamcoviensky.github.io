# Repository of Adam Coviensky's Data Science Work
##### This repository is composed of individual work, projects, and coursework from the MSc in Data Science at Columbia University


## Projects

#### Analyzing taxi supply-demand gap at LGA
Our team worked with NYC Taxi and Limousine Commission to analyze the supply and demand of taxis at LGA and produce a webapp to be used by port authority to visualize the predicted number of taxis needed at the airport for the subsequent six hours given current conditions. My main contributions were creating the ensemble model used in the web app, data cleaning and exploration, and creating the callback features of the app.

[Video of the Web App](Capstone/LGAWebApp.mov)

[Final Report](Capstone/FinalReport.pdf)

#### Creating a user interactive book recommender system
We created a web app for book recommendations from a Goodreads dataset consisting of over 6 million ratings on 10k books and 53k. We built a factorization machine which takes in user input on the weighting of genres from which they want to receive their recommendations. The model's learned parameters are then adjusted in real-time and new predictions are made based off the user input. My main contributinos were testing different factorization machine packages as well as my own, preprocessing the data, and creating the functions to make new predictions based on user input.

[Repository](https://github.com/sdoctor7/book-recommendations)

[WebApp](http://what-should-i-read-next.herokuapp.com/)

#### Analyzing up and coming neighbourhoods in New York City
Using census data as well as sidewalk cafe and bar licenses, we made visualizations to perform exploratory analysis on which regions in New York City are "up and coming."

[Executive Summary](https://marikalohmus.shinyapps.io/executive_summary/)

## Applied Machine Learning

#### Predicting NYC apartment listing prices 
I built a regression model to predict optimal listing prices of NYC apartments. Using survey data consisting of over 15k samples and 80 features. I performed both manual and automatic feature selection, feature engineering, and tuned the model's hyper parameters to achieve R^2 above 0.60.

[Python Code](AML/homework2_rent.py)

#### Predicting Bank Campaign Subscriptions
A banking institution ran a direct marketing campaign based on phone calls. We aim to predict whether someone will subscribe to the term deposit or not based on the given personal information and information from previous contacts . We approach the problem using simple machine learning models and then testing different ensemble methods to improve model results.

[Jupyter Notebook](https://nbviewer.jupyter.org/urls/adamcoviensky.github.io/AML/Clean_HW2.ipynb)


