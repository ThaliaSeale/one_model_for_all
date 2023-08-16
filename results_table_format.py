import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_table = pd.read_csv('results_archive/results_table.csv')
results_table = results_table.sort_values(by='limited',ascending=False)
model_order = ['U-Net trained from scratch', 'Finetuned U-Net', 'Progressive U-Net']
results_table['model'] = pd.Categorical(results_table['model'], categories=model_order, ordered=True)
dataset_order = ['BRATS', 'ATLAS', 'ISLES', 'MSSEG', 'TBI', 'WMH']
results_table['dataset'] = pd.Categorical(results_table['dataset'], categories=dataset_order, ordered=True)
results_table['best_smoothed_val'] = pd.to_numeric(results_table['best_smoothed_val'])
results_table = results_table.pivot(index=['dataset'],columns=['limited','model'],values=['best_smoothed_val'])

print(results_table.to_latex(float_format="{%.4f}"))

# sns.set(style="whitegrid")
# plt.figure(figsize=(15, 6))

# sns.barplot(data=results_table.loc[results_table['limited']==False], x="dataset", y="best_smoothed_val", hue="model",
#             palette=['#bc5090','#ffa600','#003f5c']).set(title='Comparison of Model Performances on Full Data')

# plt.xlabel('Dataset')
# plt.ylabel('Best Moving Avg DICE on Validation Set')
# plt.legend(title='Model', loc='lower right',framealpha=0.95)
# plt.title('Model Performance on Full Data')
# # plt.show()
# plt.savefig('results_archive/full_data.pdf')
# plt.close()

# sns.set(style="whitegrid")
# plt.figure(figsize=(15, 6))

# sns.barplot(data=results_table.loc[results_table['limited']==True], x="dataset", y="best_smoothed_val", hue="model",
#             palette=['#bc5090','#ffa600','#003f5c']).set(title='Comparison of Model Performances on Limited Data')
# plt.xlabel('Dataset')
# plt.ylabel('Best Moving Avg DICE on Validation Set')
# plt.legend(title='Model', loc='lower right',framealpha=0.95)
# plt.title('Model Performance on Limited Data')
# # plt.show()
plt.savefig('results_archive/limited_data.pdf')