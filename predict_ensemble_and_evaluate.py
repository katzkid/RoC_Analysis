import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.base import BaseEstimator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import pickle



def predict_ensemble_and_evaluate(list_folds_best_models, test_loader):
    """
    Creates one large ensemble from all models across all folds, then
    generates the full ROC curves for both soft and hard voting.

    Args:
        list_folds_best_models (List[List[Dict]]):
            A list of lists. Each inner list represents a fold and contains
            dictionaries for the best models from that fold.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        dict: A dictionary containing the ROC curve data for each voting method.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Part 1: Prepare Data and Models ---

    # Flatten the list of lists into a single list containing all models
    # This is the key change to include all models in the ensemble.
    all_models_flat = [
        model_dict
        for fold in list_folds_best_models
        for model_dict in fold
    ]

    if not all_models_flat:
        print("Error: No models found after processing the input list.")
        return None

    print(f"Creating a single ensemble from {len(all_models_flat)} models across all folds.")

    # Extract the full test set into NumPy arrays for scikit-learn compatibility
    print("Extracting full dataset...")
    X_test_list, y_test_list = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            X_test_list.append(inputs.cpu().numpy().reshape(len(inputs), -1))
            y_test_list.append(labels.cpu().numpy())
    X_test = np.vstack(X_test_list)
    true_labels = np.concatenate(y_test_list).flatten()


    # --- Part 2: Get Predictions from Every Model ---
    print("Getting predictions from all models...")
    all_probas = []
    for item in all_models_flat:
        model = item['model']
        fold_probas = None

        # --- PyTorch Model Logic ---
        if isinstance(model, nn.Module):
            model.to(device)
            model.eval()
            probas_list = []
            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probas = torch.sigmoid(outputs)
                    probas_list.extend(probas.view(-1).cpu().numpy())
            fold_probas = np.array(probas_list)

        # --- Scikit-learn Model Logic ---
        elif isinstance(model, BaseEstimator) and hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_test)
            fold_probas = probas[:, 1]
        else:
            print(f"Warning: Skipping model of unsupported type: {type(model)}")
            continue

        all_probas.append(fold_probas)

    all_probas = np.array(all_probas)
    results = {}

    # --- Part 3: Soft Voting (Averaging all model probabilities) ---
    soft_vote_probas = np.mean(all_probas, axis=0)
    fpr_sv, tpr_sv, thresholds_sv = roc_curve(true_labels, soft_vote_probas)
    results['soft_voting'] = [{'fpr': f, 'tpr': t, 'threshold': th} for f, t, th in zip(fpr_sv, tpr_sv, thresholds_sv)]

    # --- Part 4: Hard Voting (Summing votes from all models) ---
    all_votes = []
    for i, probas in enumerate(all_probas):
        threshold = all_models_flat[i]['threshold']
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.item()
        votes = (probas >= threshold).astype(int)
        all_votes.append(votes)

    hard_vote_scores = np.sum(np.array(all_votes), axis=0)
    fpr_hv, tpr_hv, thresholds_hv = roc_curve(true_labels, hard_vote_scores)
    results['hard_voting'] = [{'fpr': f, 'tpr': t, 'threshold': th} for f, t, th in zip(fpr_hv, tpr_hv, thresholds_hv)]

    return results



def plot_roc_comparison(results_lists, names, results_original_roc):
    """
    Creates a plot comparing the performance of multiple classifier sets.
    Each set's performance is shown as a connected line of points.
    The last line in the plot is highlighted to be thicker and dashed.

    Args:
        results_lists (list): A list of lists of dictionaries. Each inner list contains
                              dictionaries with 'fpr' and 'tpr' keys.
        names (list): A list of strings, where each name corresponds to a list in results_lists.
        results_original_roc (dict): A dictionary for the baseline ROC curve, containing:
                                     - "name" (str): The name of the original curve.
                                     - "auc" (float): The pre-calculated AUC score.
                                     - "fpr" (array-like): The false positive rates.
                                     - "tpr" (array-like): The true positive rates.
    """
    # --- Input Validation ---
    if not results_lists or not names:
        print("No results or names provided to plot.")
        return
    if len(results_lists) != len(names):
        print("Error: The number of result lists must match the number of names.")
        return

    # --- Plotting Setup ---
    plt.figure(figsize=(11, 11))
    colors = ['red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    num_lists = len(results_lists)

    

    # --- Plot each performance set ---
    for i, results_list in enumerate(results_lists):
        name = names[i]

        if not results_list:
            print(f"Skipping '{name}' as its result list is empty.")
            continue
            
        # Convert results to a pandas DataFrame for easy sorting
        df = pd.DataFrame(results_list)
        if 'fpr' not in df.columns or 'tpr' not in df.columns:
            print(f"Skipping '{name}' due to missing 'fpr' or 'tpr' keys in its data.")
            continue
            
        df_sorted = df.sort_values(by='fpr').reset_index(drop=True)
        
        # Select a color for this set
        color = colors[i % len(colors)]

        linewidth = 1.5
        linestyle = '-'
        alpha_value = 0.8
        plot_zorder =  3 # zorder brings the line to the front

        # reset plot parameters
        is_soft = False
        is_hard = False
        
        # Calculate the Area Under the Curve for the connected points
        roc_auc = auc(df_sorted["fpr"], df_sorted["tpr"])

        # --- MODIFICATION START ---
        # Check if the current list is the last one
        is_soft = (name == "Ensemble_voting_soft")

        is_hard = (name == "Ensemble_voting_hard")

        if is_soft or is_hard:
            # Set distinctive styles for the last line, and standard for others
            linewidth = 3.0 
            linestyle = '--' 
            alpha_value = 0.9 
            plot_zorder = 5 
            if is_soft:
                color = 'red'
            else:
                color = 'green'
        # --- MODIFICATION END ---

        # Plot the line connecting the points for this set
        plt.plot(df_sorted['fpr'], df_sorted['tpr'], color=color, lw=linewidth,
                 linestyle=linestyle, alpha=alpha_value, zorder=plot_zorder,
                 label=f'{name} (AUC = {roc_auc:.2f})')

        # Plot the individual model points as a scatter plot
        plt.scatter(df_sorted['fpr'], df_sorted['tpr'], c=color, marker='o', 
                    alpha=0.6, s=80, zorder=plot_zorder + 1) # Place scatter points on top of the line

    # --- Plot the original ROC curve for reference ---
    if results_original_roc:
        plt.plot(results_original_roc["fpr"], results_original_roc["tpr"], color='blue', lw=2.5,
                 label=f'{results_original_roc["name"]} (AUC = {results_original_roc["auc"]:.2f})')

    # --- Final plot styling ---
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # "No-skill" line
    
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()



def make_curve_monotonic(points):
    """
    Processes a list of ROC points to ensure the TPR is monotonically increasing.

    Args:
        points (List[Dict]): A list of dictionaries, each with 'fpr' and 'tpr' keys.

    Returns:
        List[Dict]: A new list of points representing a monotonic ROC curve.
    """
    if not points:
        return []

    # Sort points primarily by FPR, then by TPR as a tie-breaker
    sorted_points = sorted(points, key=lambda p: (p['fpr'], p['tpr']))

    monotonic_points = [sorted_points[0]]
    for point in sorted_points[1:]:
        # Get the last point added to our clean list
        last_monotonic_point = monotonic_points[-1]

        # Keep the new point only if it has a higher or equal TPR.
        # This creates the "upper envelope" of the raw curve.
        if point['tpr'] >= last_monotonic_point['tpr']:
            monotonic_points.append(point)

    return monotonic_points

def save_to_pickle(list_folds_best_models, list_folds_weighted_clfs, results_original_roc, test_loader, filename=''):
    

    # 1. Group all objects into a single dictionary
    data_to_save = {
        'best_models': list_folds_best_models,
        'weighted_clfs': list_folds_weighted_clfs,
        'roc_results': results_original_roc,
        'test_loader': test_loader
    }

    # 2. Save the dictionary to a file
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Data saved to {filename}")

    return data_to_save

def load_from_pickle(filename=''):
    # Load the dictionary from the file
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)

    # Extract your variables
    list_folds_best_models = loaded_data['best_models']
    list_folds_weighted_clfs = loaded_data['weighted_clfs']
    results_original_roc = loaded_data['roc_results']
    test_loader = loaded_data['test_loader']

    print("Data loaded successfully.")

    return list_folds_best_models, list_folds_weighted_clfs, results_original_roc, test_loader