import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import sys
import math

# Step 1: Load TensorBoard logs and extract data
# Replace 'path/to/tensorboard/logs/' with the actual path to your TensorBoard logs
# In this example, 'loss' and 'accuracy' are scalars logged during training.

ISLES_results_dir = "results_archive/ISLES_FINAL"
BRATS_results_dir = "results_archive/BRATS_FINAL"
ATLAS_results_dir = "results_archive/ATLAS_FINAL"
TBI_results_dir = "results_archive/TBI_FINAL"
WMH_results_dir = "results_archive/WMH_FINAL"
MSSEG_results_dir = "results_archive/MSSEG_FINAL"

def extract_results(results_path):
    data = {'step': [], 'accuracy': [], 'model': [], 'limited': [], 'run': []}

    search_pattern = r'^val_mean_dice'

    for root, _, files in os.walk(results_path):
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
                                else:
                                    data['model'].append('Progressive U-Net')
                            else:
                                data['model'].append('U-Net trained from scratch')
                            data['limited'].append(re.search(r'limited', file_path) is not None)
                            data['run'].append(re.findall(r'\d+\.\d+\.\d+\-\d+\.\d+\.\d+',file_path)[0])
    data = pd.DataFrame(data)
    return(data)                       

results_table = pd.DataFrame(columns = ['dataset','model','limited','run','best_smoothed_val'])

def smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed

def append_results(dataset,results_path,results_table):
    results = extract_results(results_path)
    for model in results['model'].unique():
        for limited in results['limited'].unique():
            for run in results['run'].unique():
                df = results.loc[(results['run'] == run) & (results['model'] == model) & (results['limited'] == limited)]
                df['accuracy_smoothed'] = smooth(df['accuracy'],0.7) 
                best_smoothed_val = df['accuracy_smoothed'].max()
                results_table = results_table.append({'dataset': dataset,
                                                      'model': model,
                                                      'limited': limited, 
                                                      'run': run,
                                                      'best_smoothed_val': best_smoothed_val}, ignore_index=True)

    # for run in results['run'].unique():
    #     df = results[results['run'] == run]
    #     df['accuracy_smoothed'] = smooth(df['accuracy'],0.7) 
    #     best_smoothed_val = df['accuracy_smoothed'].max()
    #     best_smoothed_val_step = df[df['accuracy_smoothed'] == best_smoothed_val]['step'].iloc[0]
    #     results_table = results_table.append({'dataset': dataset,
    #                                           'model': df['model'].iloc[0],
    #                                           'limited': df['limited'].iloc[0],
    #                                           'run': run,
    #                                           'best_smoothed_val': best_smoothed_val}, ignore_index=True)

    results_table = results_table.dropna()
    return(results_table)

for dataset in ['BRATS','ATLAS','TBI','WMH','MSSEG','ISLES']:
    results_table = append_results(dataset,globals()[dataset + '_results_dir'],results_table)

results_table = results_table.drop(columns = ['run'])

results_table = results_table.groupby(['dataset','model','limited']).mean()

results_table.to_csv('results_archive/results_table.csv')
# # Step 2: Create a Pandas DataFrame
# df = pd.DataFrame(data)

# # Step 3: Convert 'step' column to numeric
# df['step'] = pd.to_numeric(df['step'])

# # Step 4: Create a Seaborn plot with title and legend
# sns.set(style="whitegrid")
# plt.figure(figsize=(15, 6))

# # Adjust the following plot based on your needs
# custom_order = ['U-Net trained from scratch', 'Finetuned U-Net', 'Progressive U-Net']
# df['model'] = pd.Categorical(df['model'], categories=custom_order, ordered=True)
# df = df.sort_values(by='model')
# for model in df['model'].unique():
#     if model == 'Finetuned U-Net':
#        factor = 2
#        color = '#bc5090'
#     elif model == 'Progressive U-Net':
#        factor = 1
#        color = '#ffa600'
#     else:
#        factor = 1
#        color = '#003f5c'
#     df_model = df[df['model'] == model]
#     df_model = df_model[df_model['limited'] == limited]
#     df_model['step'] = df_model['step'] * factor
#     df_model['accuracy_smoothed'] = df_model['accuracy'].ewm(alpha=0.7).mean()
#     sns.lineplot(x='step', y='accuracy', data=df_model, label=model, alpha = 0.25, color = color)
#     sns.lineplot(x='step', y='accuracy_smoothed', data=df_model, label=model + " (Smoothed)",color = color)

# if limited:
#     plt.title('Model Performance on ' + dataset + ' (Limited Dataset)')
# else:
#     plt.title('Model Performance on ' + dataset + ' (Full Dataset)')
# plt.xlabel('Step')
# plt.ylabel('Avg DICE on Validation Set')
# plt.legend(title='Model', loc='best')

# # plt.show()

# plt.savefig(logs_dir + '/' + dataset + '_plot.pdf')