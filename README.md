# Project: Empathy Prediction using Eye-Tracking Data

This Python project aims to predict empathy levels in individuals based on eye-tracking data. The project utilizes machine learning techniques, including RandomForestRegressor, to build a predictive model.

### Getting Started
These instructions will guide you on how to set up the project on your local machine for development and testing purposes.

### Prerequisites
To run this project, you'll need Python 3.x and the following libraries installed:

pandas
numpy
matplotlib
seaborn
scikit-learn
pickle
You can install these libraries using the following command:

Copy code
pip install pandas numpy matplotlib seaborn scikit-learn pickle
Project Structure
The project is organized as follows:

diff
Copy code
- empathyhelper.py: Helper functions for data preprocessing, feature extraction, and model evaluation
- output_data.csv: Directory containing the raw eye-tracking data files (CSV format)
Usage
To run the project, simply execute the main_empathy.py script. The script will preprocess the data, extract relevant features, train the RandomForestRegressor model, and evaluate the model's performance using cross-validation.

Modify this template to fit your project's specific needs and requirements. Make sure to include all the necessary details to help users understand the project and its functionality.




# Week 2 - Summary and resampling statistics

One of the core concepts we learnt in the lecture this week was that of resampling statistics, and we placed emphasis on the bootstrap as a very useful tool (that we will use in some of the next lectures as well).

In this lab, you will use Numpy to create your own bootstrap and permutation test functions in Python.
For this, you have some starting code and instructions on the notebook _statistics.ipynb_.

Once you have the two functions working, go to Moodle to start the quiz for this week.
As last week, your answers will be automatically marked.

* rnn.ipynb contains the code shown in the lecture 
* weather.csv contains the weather dataset used in the lecture

Other files for this week:

- _customers.csv_: dataset of sales used to test your bootstrap function.
- _vehicles.csv_: dataset of vehicle consumption (in mpg -- miles per gallon) for two different fleets -- you'll need this for the quiz.
- _voting\_data.py_: dataset of percentages of democratic votes in Pennsylvania and Ohio -- you'll need this for the quiz.
