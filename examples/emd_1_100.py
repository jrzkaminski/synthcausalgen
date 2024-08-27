import os
import pandas as pd
import numpy as np
import ot
import matplotlib.pyplot as plt

# Path to the dataset and DAG folder
folder_path = 'large_dag_data'

# List all files in the folder
files = os.listdir(folder_path)

# Separate DAG and dataset files, sorted by n
dag_files = sorted([f for f in files if 'dag_' in f], key=lambda x: int(x.split('_')[1].split('.')[0]))
dataset_files = sorted([f for f in files if 'dataset_' in f], key=lambda x: int(x.split('_')[1].split('.')[0]))


def compute_emd(data1, data2):
    # Convert to numpy arrays (ensure same size)
    data1 = data1.values
    data2 = data2.values

    # Normalize data if needed
    data1 = data1 / np.sum(data1)
    data2 = data2 / np.sum(data2)

    # Calculate EMD
    emd_value = ot.emd2([], [], data1, data2)
    return emd_value


# Load dag_0 and dataset_0
dag_0 = pd.read_csv(os.path.join(folder_path, dag_files[0]))
dataset_0 = pd.read_csv(os.path.join(folder_path, dataset_files[0]))

# List to store EMD values
emd_values = []

# Compute EMD for each n from 1 to 100
for i in range(1, 101):
    dag_n = pd.read_csv(os.path.join(folder_path, dag_files[i]))
    dataset_n = pd.read_csv(os.path.join(folder_path, dataset_files[i]))

    # Compute EMD between dataset_0 and dataset_n
    emd_dataset = compute_emd(dataset_0, dataset_n)

    # Average EMD for dag and dataset (if necessary)
    emd_values.append(emd_dataset)

print(emd_values)
# Generate the plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), emd_values, marker='o', linestyle='-', color='b')
plt.title('EMD between dag_0/dataset_0 and dag_n/dataset_n (n=1 to 100)')
plt.xlabel('n')
plt.ylabel('EMD')
plt.grid(True)
plt.show()