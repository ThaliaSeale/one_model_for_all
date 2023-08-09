import tensorflow as tf

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import sys

# Step 1: Load TensorBoard logs and extract data
# Replace 'path/to/tensorboard/logs/' with the actual path to your TensorBoard logs
# In this example, 'loss' and 'accuracy' are scalars logged during training.

limited = bool(int(sys.argv[1])) 
dataset = str(sys.argv[2])

logs_dir = str(sys.argv[3])

data = {'step': [], 'accuracy': [], 'model': [], 'limited': []}

search_pattern = r'^val_mean_dice'

for root, _, files in os.walk(logs_dir):
    for file in files:
        if file.startswith("events.out.tfevents"):
            file_path = os.path.join(root, file)
            for event in tf.compat.v1.train.summary_iterator(file_path):
                for value in event.summary.value:
                    if re.match(search_pattern, value.tag):
                        data['accuracy'].append(value.simple_value)
                        data['step'].append(event.step)
                        if re.search(r'pretrained', file_path):
                            if re.search(r'naive', file_path):
                                data['model'].append('Finetuned U-Net')
                            elif re.search(r'reduced', file_path):
                                data['model'].append('Reduced U-Net')
                            else:
                                data['model'].append('Progressive U-Net')
                        else:
                            data['model'].append('U-Net trained from scratch')
                        data['limited'].append(re.search(r'limited', file_path) is not None)

# Step 2: Create a Pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Convert 'step' column to numeric
df['step'] = pd.to_numeric(df['step'])

# Cropping the training curve
# df = df.loc[df['step']<=35000]
df = df.loc[~df['model'].isin(['Reduced U-Net'])]
# Step 4: Create a Seaborn plot with title and legend
sns.set(style="whitegrid")
plt.figure(figsize=(15, 6))

# Adjust the following plot based on your needs
custom_order = ['U-Net trained from scratch', 'Finetuned U-Net', 'Progressive U-Net', 'Reduced U-Net']
df['model'] = pd.Categorical(df['model'], categories=custom_order, ordered=True)
df = df.sort_values(by='model')
for model in df['model'].unique():
    if model == 'Finetuned U-Net':
       factor = 1
       color = '#bc5090'
    elif model == 'Progressive U-Net':
       factor = 1
       color = '#ffa600'
    elif model == 'Reduced U-Net':
        factor = 1
        color = "#7a5195"
    else:
       factor = 1
       color = '#003f5c'
    df_model = df[df['model'] == model]
    df_model = df_model[df_model['limited'] == limited]
    df_model['step'] = df_model['step'] * factor
    df_model['accuracy_smoothed'] = df_model['accuracy'].ewm(alpha=0.7).mean()
    sns.lineplot(x='step', y='accuracy', data=df_model, label=model, alpha = 0.25, color = color)
    sns.lineplot(x='step', y='accuracy_smoothed', data=df_model, label=model + " (Smoothed)",color = color)

if limited:
    plt.title('Model Performance on ' + dataset + ' (Limited Dataset)')
else:
    plt.title('Model Performance on ' + dataset + ' (Full Dataset)')
plt.xlabel('Step')
plt.ylabel('Avg DICE on Validation Set')
plt.legend(title='Model', loc='best')

# plt.show()

plt.savefig(logs_dir + '/' + dataset + '_training_plot' + str(limited) +  '.pdf')
