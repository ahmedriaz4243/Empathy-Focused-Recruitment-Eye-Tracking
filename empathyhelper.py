
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import glob


from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, GroupKFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score

import pickle

warnings.filterwarnings("ignore")



def preprocess_data(data):
    # Drop the first column
    data = data.iloc[:, 1:]
    
    # List of columns to drop if they exist not important for Empathy
    cols_to_drop = ['Mouse position X', 'Mouse position Y', 'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)',
                    'Event', 'Event value',
                    'Computer timestamp', 'Export date', 'Recording date',
                    'Recording date UTC', 'Recording start time', 'Timeline name', 'Recording Fixation filter name',
                    'Recording software version', 'Recording resolution height', 'Recording resolution width',
                    'Recording monitor latency', 'Presented Media width', 'Presented Media height',
                    'Presented Media position X (DACSpx)', 'Presented Media position Y (DACSpx)', 'Original Media width',
                    'Recording start time UTC', 'Original Media height', 'Sensor']


    # Forward fill the pupil diameter and fixation point columns
    data[['Pupil diameter left', 'Pupil diameter right', 'Fixation point X', 'Fixation point Y']] = \
        data[['Pupil diameter left', 'Pupil diameter right', 'Fixation point X', 'Fixation point Y']].ffill()

    # List of columns to be converted to numerical values
    num_cols = ['Gaze direction left X', 'Gaze direction left Y', 'Gaze direction left Z',
                'Gaze direction right X', 'Gaze direction right Y', 'Gaze direction right Z',
                'Eye position left X (DACSmm)', 'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)',
                'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)', 'Eye position right Z (DACSmm)',
                'Gaze point left X (DACSmm)', 'Gaze point left Y (DACSmm)', 'Gaze point right X (DACSmm)',
                'Gaze point right Y (DACSmm)', 'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)',
                'Gaze point left X (MCSnorm)', 'Gaze point left Y (MCSnorm)', 'Gaze point right X (MCSnorm)',
                'Gaze point right Y (MCSnorm)', 'Pupil diameter left', 'Pupil diameter right']

    # Convert the string values into numbers
    for col in num_cols:
        data[col] = pd.to_numeric(data[col].str.replace(',', '.'), errors='coerce')

   
   
    return data
    


def summarize_eye_tracking_data(data, group):
    # Filter valid gaze data
    valid_data = data[(data['Validity left'] == 'Valid') & (data['Validity right'] == 'Valid')]

    # Total fixations
    total_fixations = data[data['Eye movement type'] == 'Fixation'].shape[0]

    # Average fixation duration
    avg_fixation_duration = data[data['Eye movement type'] == 'Fixation']['Gaze event duration'].mean()

    # Calculate mean, median, and std of pupil diameter, Gaze point X, Gaze point Y, Fixation point X, and Fixation point Y
    pupil_diameter_stats = data[['Pupil diameter left', 'Pupil diameter right']].mean(axis=1).agg(['mean', 'median', 'std']).rename(lambda x: f'Pupil Diameter {x.capitalize()}')
    gaze_point_x_stats = data['Gaze point X'].agg(['mean', 'median', 'std']).rename(lambda x: f'Gaze Point X {x.capitalize()}')
    gaze_point_y_stats = data['Gaze point Y'].agg(['mean', 'median', 'std']).rename(lambda x: f'Gaze Point Y {x.capitalize()}')
    fixation_point_x_stats = data['Fixation point X'].agg(['mean', 'median', 'std']).rename(lambda x: f'Fixation Point X {x.capitalize()}')
    fixation_point_y_stats = data['Fixation point Y'].agg(['mean', 'median', 'std']).rename(lambda x: f'Fixation Point Y {x.capitalize()}')

    # Create summary row
    summary_data = {
        'Participant Name': data['Participant name'].iloc[0],
        'Project Name': group,
        'Recording Name': data['Recording name'].iloc[0],
        'Total Fixations': total_fixations,
        'Avg. Fixation Duration': avg_fixation_duration
    }
    summary_data.update(pupil_diameter_stats)
    summary_data.update(gaze_point_x_stats)
    summary_data.update(gaze_point_y_stats)
    summary_data.update(fixation_point_x_stats)
    summary_data.update(fixation_point_y_stats)

    summary = pd.DataFrame(summary_data, index=[0])

    return summary


def uncertainty_diameters(result_I):
    # Get unique participant names
    unique_participants = result_I['Participant Name'].unique()

    # Iterate over each unique participant
    for participant in unique_participants:
        
        participant_data = result_I[result_I['Participant Name'] == participant]

        
        participant_data = participant_data.reset_index().rename(columns={'index': 'occurrence'}).head(6)

        # Group by occurrence
        grouped_data = participant_data.groupby('occurrence').agg({'Pupil Diameter Mean': 'mean', 'Pupil Diameter Median': 'mean', 'Pupil Diameter Std': 'mean'}).reset_index()

        
        fig, ax = plt.subplots()

        
        ax.errorbar(grouped_data['occurrence'], grouped_data['Pupil Diameter Mean'], grouped_data['Pupil Diameter Std'], linestyle='-', marker='o', capsize=5, ecolor="green", elinewidth=0.5, label='Mean')

        
        ax.plot(grouped_data['occurrence'], grouped_data['Pupil Diameter Median'], linestyle='-', marker='s', label='Median')

        
        ax.set_xlabel('Occurrence')
        ax.set_ylabel('Avg Pupil Diameter (mm)')
        ax.set_title(f'Mean and Median, {participant}')

        
        ax.legend()

        
        plt.show()
        
        
        return
        

    
    

def plot_actual_vs_predicted(dataframe):
    y_test_all = dataframe['Original Empathy Score'].tolist()
    y_pred_all = dataframe['Predicted Empathy Score'].tolist()

    plt.scatter(y_test_all, y_pred_all, color='blue', label='Predicted')
    plt.xlabel('Original Empathy Scores')
    plt.ylabel('Predicted Empathy Scores')
    plt.title('Original vs. Predicted Empathy Scores ')

    # Add a perfect prediction line
    min_val = min(min(y_test_all), min(y_pred_all))
    max_val = max(max(y_test_all), max(y_pred_all))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', label='Perfect Prediction')

    plt.legend()
    plt.show()
    
    return


def train_and_evalute(data_df, group_name):
    X = data_df.drop(columns=['Total Score extended', 'Project Name', 'Recording Name'])
    y = data_df['Total Score extended']
    results_df = pd.DataFrame(columns=['Participant Name', 'Original Empathy Score', 'Predicted Empathy Score'])
    # Encode the 'Participant Name' column
    encoder = LabelEncoder()
    X['Participant Name'] = encoder.fit_transform(X['Participant Name'])
    groups = data_df['Participant Name']  
    
    n_splits = 30  # No of Participant
    gkf = GroupKFold(n_splits=n_splits)
    
    mse_scores = []
    r2_scores = []
    rt_MSE_scores = []
    medae_scores = []
    y_test_all = []  # Initialize y_test_all list
    y_pred_all = []  # Initialize y_pred_all list
    
    for fold, (train_index, test_index) in enumerate(gkf.split(X, y, groups=groups)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    
        ###################################
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        # Make predictions and evaluate the model
        y_pred = model.predict(X_test)
        
        ###################################### 
    
        
    
        print(f"Fold {fold + 1}:")
    
        for idx, (original, predicted) in enumerate(zip(y_test, y_pred)):
            participant_name = data_df.iloc[test_index[idx]]['Participant Name']
            print(f"  Participant Name: {participant_name}, Orignal Empathy Score: {original}, Predicted empathy Score: {predicted:.2f}")
            results_df = results_df.append({'Participant Name': participant_name,'Original Empathy Score': original,'Predicted Empathy Score': predicted}, ignore_index=True)
   
    
        mse = mean_squared_error(y_test, y_pred)
        rt_MSE = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
    
    
    
        mse_scores.append(mse)
        r2_scores.append(r2)
        rt_MSE_scores.append(rt_MSE)
        medae_scores.append(medae)
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)
    

        # Calculate the average MSE and R2 scores
    avg_r2 = np.mean(r2_scores)
    avg_rt = np.mean(rt_MSE_scores)
    avg_medae = np.mean(medae_scores)
    avg_mse_scores = np.mean(mse_scores)


    print(f"Average R2: {avg_r2}")
    print(f"Average Rt mean sqr error: {avg_rt}")
    print(f"Average Median Abs error: {avg_medae}")
    print(f"Average Mean Squared Error: {avg_mse_scores}")  # Corrected this line



    return results_df


def plot_correlation_heatmap(df, target_column, top_n=15):
    # Compute the correlation matrix for the dataframe
    corr_matrix = df.corr()

    # Select the top_n columns with the highest correlation
    cols = corr_matrix.nlargest(top_n, target_column)[target_column].index

    # Compute the correlation matrix for the selected columns
    cm = df[cols].corr()

    # Set the size of the heatmap
    plt.figure(figsize=(10, 10))

    # Draw the heatmap with annotations
    ax = sns.heatmap(cm, annot=True, cmap='coolwarm')
    ax.set_title('Correlation matrix for ' + target_column)

    # Show the plot
    plt.show()
    
    return