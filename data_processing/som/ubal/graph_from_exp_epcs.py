import pickle
from tabulate import tabulate

# Load the pickled data
with open('results_all.pkl', 'rb') as f:
    results_dict = pickle.load(f)

# Prepare data for the table
table_data = []
for epoch, results in results_dict.items():
    train_mean, train_stdev = results['train']
    test_mean, test_stdev = results['test']
    table_data.append([epoch, train_mean, train_stdev, test_mean, test_stdev])

# Define headers for the table
headers = ['Epochs', 'Train Mean (%)', 'Train Stdev (%)', 'Test Mean (%)', 'Test Stdev (%)']

# Generate LaTeX table
latex_table = tabulate(table_data, headers=headers, tablefmt='latex_raw')

# Output LaTeX code
print(latex_table)
