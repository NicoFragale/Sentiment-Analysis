# Introductions

This project focuses on creating a machine learning model 
capable of determining the sentiment(positive or negative) of textual reviews.

# Workflow analysis and functions:

1. Data collection and preparation:
Obtain a suitable dataset for training, validation, and testing.

2. Tokenization and Padding:
Convert text into uniform numeric sequences so that they are valid input
for the model.

3. Model Building and Training:
Create and configure the neural network model using embedding, convo-
lution and pooling techniques.

4. Validation and Testing:
Optimize of parameters to increase model accuracy.

5. Analyze and Save the Model:
Save the trained model to document the results obtained and for future
use.

6. Model Application:
Apply the trained model on new data to ensure it works correctly.

    (a) Model application on a dataset:
    Apply the model on a dataset saved locally in your computer.

    (b) Model application on the scraping result:
    Apply the model on the results of a web scaper on the rotten tomatoes
    comments section.

# Conclusions

In summary, this project represents both a significant step towards auto-
mation and efficiency in sentiment analysis and provides a solid foundation
for further developments and innovations in the field of data analytics. The
integration of web scraping for dynamic data acquisition further enhances the
system’s utility, making it a comprehensive tool for sentiment analysis and
customer feedback management.

## How to run the programm
1. Install the resources specified below;
2. Create conda enviroment and install all the libraries;
3. Run main.py;
4. Use GUI as you prefer.

### How to install the envirorment. 

## Open Anaconda Navigator.

## Go to the "Environments" Section: Click on "Environments" in the left-hand panel.

## Import Environment:

1. Click the "Import" button located in the lower central part of the environments screen.
2. Select the `environment.yml` file that you exported.
3. Assign a name to the new environment (it can be the same as the old environment or a new one).

## Create Environment:  Click "Import" to create the new environment based on the `environment.yml` file.


## EXAMPLE, Do you want to know if Avengers: Endgame is a good film?
1. Run main.py
2. Select "Analisi del sentiment dei commenti di Rotten Tomatoes"
3. Seclect from the "models" directory the model you want to apply to the reviews and its tokenizer and padding infos
4. Insert the URL, in this case : (https://www.rottentomatoes.com/m/avengers_endgame/reviews)
5. Let the model run and wait for the response!

## Resources
1. glove.42B.300d.txt is embedding used to train the model; [link to install](https://www.kaggle.com/datasets/yutanakamura/glove42b300dtxt)
2. rotten_tomatoes_movie_reviews.csv used to train the model; [link to install](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews/data)
3. IMDB Dataset.csv used to apply the model; [link to install](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
)


ENJOY!