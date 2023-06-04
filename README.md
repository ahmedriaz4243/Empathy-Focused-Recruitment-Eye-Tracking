# Project: Empathy Prediction using Eye-Tracking Data

This Python project aims to predict empathy levels in individuals based on eye-tracking data. The project utilizes machine learning techniques, including ####RandomForestRegressor with ####GroupKfold validation, to build a predictive model.

### Getting Started
These instructions will guide you on how to set up the project on your local machine for development and testing purposes.

### Prerequisites
To run this project, you'll need Python 3.x and the following libraries installed:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* pickle

###### You can install these libraries using the following command:

pip install pandas numpy matplotlib seaborn scikit-learn pickle

### Project Structure
The project is organized as follows:

- empathyhelper.py: Helper functions for data preprocessing, feature extraction, and model evaluation
- EyeT4Empathy-Dataset-analysis.ipynb: is the ML pipeline with Exploration and Example.

* Usage
To run the project, simply execute the EyeT4Empathy-Dataset-analysis.ipynb. This will preprocess the data, extract relevant features, train the RandomForestRegressor model, and evaluate the model's performance using cross-validation.

Make sure to include all the necessary details to help users understand the project and its functionality.

### Dataset Acknowledgements

The following data is used in this study. To utilize the dataset, simply download it and adjust the file paths according to your setup.

I would like to express my gratitude to the authors and contributors of the EyeT4Empathy dataset for making it publicly available. The dataset can be found at the following link:

[EyeT4Empathy Dataset](https://doi.org/10.1038/s41597-022-01862-w)

Please cite the dataset using the following reference:

P. Lencastre, S. Bhurtel, A. Yazidi, S. Denysov, P. G. Lind, et al. EyeT4Empathy: Dataset of foraging for visual information, gaze typing and empathy assessment. Scientific Data, 9(1):1–8, 2022


<pre>
```bibtex
@article{Lencastre2022,
  author = {Lencastre, Pedro and Bhurtel, Sanchita and Yazidi, Anis and et al.},
  title = {EyeT4Empathy: Dataset of foraging for visual information, gaze typing and empathy assessment},
  journal = {Sci Data},
  volume = {9},
  pages = {752},
  year = {2022},
  doi = {10.1038/s41597-022-01862-w}
}
```
</pre>
