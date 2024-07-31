# Sentiment Analysis on Movie Reviews
This project performs sentiment analysis on movie reviews using various machine learning techniques. The data consists of movie reviews and metadata, with the goal of predicting the sentiment of reviews as either positive or negative.

## Project Structure
data/: Contains the dataset files used for analysis.

movies.csv: Contains movie metadata.
train.csv: Contains training data for sentiment prediction.
test.csv: Contains test data for sentiment prediction.
analysis.py: The main Python script for data loading, preprocessing, and analysis.

## Dependencies
The project requires the following Python packages:

pandas
numpy
regex
string
scikit-learn
imbalanced-learn
matplotlib
seaborn


You can install these dependencies using pip:
pip install pandas numpy regex scikit-learn imbalanced-learn matplotlib seaborn


## Data Loading
The data is loaded from CSV files into Pandas DataFrames:
movies_df = pd.read_csv('/kaggle/input/sentiment-prediction-on-movie-reviews/movies.csv')
train_df = pd.read_csv('/kaggle/input/sentiment-prediction-on-movie-reviews/train.csv')
test_df = pd.read_csv('/kaggle/input/sentiment-prediction-on-movie-reviews/test.csv')


## Data Overview
movies_df contains movie metadata with columns such as movieid, title, audienceScore, and genre.
train_df contains training data with columns such as movieid, reviewerName, isFrequentReviewer, reviewText, and sentiment.
test_df contains test data with columns such as movieid, reviewerName, isTopCritic, and reviewText.


## Data Cleaning and Preprocessing
Duplicates are removed from the movies_df and train_df DataFrames.
Missing values in the reviewText column of train_df are dropped.
Data is reset to ensure proper indexing.

movies_df.drop_duplicates(subset='movieid', inplace=True)
train_df.drop_duplicates(subset=['movieid', 'reviewerName', 'reviewText'], inplace=True)
train_df.dropna(subset='reviewText', inplace=True)
train_df.reset_index(drop=True, inplace=True)
movies_df.reset_index(drop=True, inplace=True)


## Exploratory Data Analysis
Sentiment Distribution
A pie chart is generated to visualize the distribution of positive and negative sentiments in the training dataset:

fig=plt.figure(figsize=(5,5))
colors=["skyblue",'pink']
pos=train_df[train_df['sentiment']=='POSITIVE']
neg=train_df[train_df['sentiment']=='NEGATIVE']
ck=[pos['sentiment'].count(),neg['sentiment'].count()]
legpie=plt.pie(ck,labels=["Positive","Negative"],
                 autopct ='%1.1f%%', 
                 shadow = True,
                 colors = colors,
                 startangle = 45,
                 explode=(0, 0.1))

## Next Steps
Feature Extraction: Use techniques like CountVectorizer, TfidfVectorizer, or HashingVectorizer to convert text data into numerical features.
Model Training: Train various models such as Logistic Regression, Naive Bayes, K-Nearest Neighbors, and Support Vector Machine on the training data.
Evaluation: Evaluate the models using accuracy, precision, recall, and F1-score metrics.

## Usage
Run the analysis.py script to execute the data loading, preprocessing, and exploratory data analysis:
python analysis.py


## License
This project is licensed under the MIT License. See the LICENSE file for details.
