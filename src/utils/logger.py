import csv # for writing logs to a CSV file
import os # for checking if the log file exists
import pandas as pd # for creating a DataFrame to store logs

def init_csv(output_dir, fieldnames):
    parent = os.path.dirname(output_dir)
    if parent:
        os.makedirs(parent, exist_ok=True)#create the directory if it doesn't exist

    with open(output_dir, mode = 'w', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader() # write the header to the CSV file

def append_row(output_dir, row_dict):
   with open(output_dir, mode = 'a', newline = '') as f:
       writer = csv.DictWriter(f, fieldnames = row_dict.keys())
       writer.writerow(row_dict) # write a row to the CSV file

def load_results(csv_path):
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].str.lower().isin(['true', 'false']).all():
                df[col] = df[col].str.lower() == 'true' # convert to boolean
    return df

def log_training_step(log_path, timestep, mean_reward, success_rate, mean_loss):
    row = {
        'timestep': timestep,
        'mean_reward': mean_reward,
        'mean_loss': mean_loss,
        'success_rate': success_rate
    }
    if not os.path.exists(log_path):
        init_csv(log_path, fieldnames = list(row.keys()))
    append_row(log_path, row)