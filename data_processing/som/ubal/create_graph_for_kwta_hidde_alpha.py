import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('results_all_grid_hidden_layer_alphas.pkl', 'rb') as f:
    results_dict = pickle.load(f)

model_names = ["lf", "rf", "rh", "lh"]
alphas = [0.01, 0.1, 0.05, 0.5, 0.3]
hidden_layers = [100, 150, 200, 250, 300]

for model_name in model_names:
    model_results = {k: v for k, v in results_dict.items() if k[0] == model_name}

    train_accuracies = np.zeros((len(alphas), len(hidden_layers)))
    test_accuracies = np.zeros((len(alphas), len(hidden_layers)))

    for j, alpha in enumerate(alphas):
        for k, hidden_layer in enumerate(hidden_layers):
            accuracies_train = []
            accuracies_test = []
            for epoch, values in model_results[(model_name, alpha, hidden_layer)].items():
                accuracies_train.append(values['train'][0])
                accuracies_test.append(values['test'][0])

            mean_accuracy_train = np.mean(accuracies_train)
            mean_accuracy_test = np.mean(accuracies_test)

            train_accuracies[j, k] = mean_accuracy_train
            test_accuracies[j, k] = mean_accuracy_test

    fig, axs = plt.subplots(2, 1, figsize=(10, 15))

    im1 = axs[0].imshow(train_accuracies, cmap='viridis', interpolation='nearest')
    #axs[0].set_title(f'Training Accuracies - Model: {model_name}')
    axs[0].set_xlabel('Hrúbka skrytej vrstvy')
    axs[0].set_ylabel('Rýchlosť učenia (alpha)')
    axs[0].set_xticks(np.arange(len(hidden_layers)))
    axs[0].set_yticks(np.arange(len(alphas)))
    axs[0].set_xticklabels(hidden_layers)
    axs[0].set_yticklabels(alphas)
    fig.colorbar(im1, ax=axs[0], orientation='vertical')

    im2 = axs[1].imshow(test_accuracies, cmap='viridis', interpolation='nearest')
    #axs[1].set_title(f'Testovacia Presnosť - Model: {model_name}')
    axs[1].set_xlabel('Hrúbka skrytej vrstvy')
    axs[1].set_ylabel('Rýchlosť učenia (alpha)')
    axs[1].set_xticks(np.arange(len(hidden_layers)))
    axs[1].set_yticks(np.arange(len(alphas)))
    axs[1].set_xticklabels(hidden_layers)
    axs[1].set_yticklabels(alphas)
    fig.colorbar(im2, ax=axs[1], orientation='vertical')
    plt.tight_layout()
    plt.savefig(f"{model_name}_accuracies.pdf", format="pdf")
    plt.close()
