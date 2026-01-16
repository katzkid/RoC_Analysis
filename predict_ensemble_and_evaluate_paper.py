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
    # prior probability
    prior_prob = np.mean(true_labels)


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

    # misclassification risk
    mis_risk = prior_prob * (1 - tpr_hv) + (1 - prior_prob) * fpr_hv
    results['misclassification_risk'] = [{'fpr': f, 'tpr': t, 'threshold': th} for f, t, th in zip(fpr_hv, mis_risk, thresholds_hv)]

    # calculate the misclassification risk for threshold equal to exactly half the votes
    total_votes = len(all_models_flat)
    th_05 = total_votes/2

    majority_preds = (hard_vote_scores >= th_05).astype(int)

    tn, fp, fn, tp = confusion_matrix(true_labels, majority_preds).ravel()

    tpr_half = tp/(tp + fn) if(tp + fn) > 0 else 0.0
    fpr_half = fp/(fp + tn) if(fp + tn) > 0 else 0.0

    risk_05 = prior_prob * (1 - tpr_half) + (1 - prior_prob) * (fpr_half)

    results['misclassification_risk_half'] = {
        'risk' : risk_05,
        'tpr' : tpr_half,
        'fpr' : fpr_half,
        'threshold' : th_05
    }

    return results, prior_prob



def plot_roc_comparison(results_lists, names, results_original_roc, plot_name = "No name specified", prior_prob=0.5, misclassification_risk=None):
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
    linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 5))),
     ('densely dotted',        (0, (1, 1))),

     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

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
        marker='+'

        # reset plot parameters
        is_soft = False
        is_hard = False
        
        # Calculate the Area Under the Curve for the connected points
        roc_auc = auc(df_sorted["fpr"], df_sorted["tpr"])

        # --- MODIFICATION START ---
        
        is_weighted = "Weighted" in name
        if is_weighted:
            num_shades = 5  # Number of shades for the weighted classifiers
            marker = '^'  # Change marker for weighted classifiers to triangle
            colors2 = [str(g) for g in np.linspace(0.2, 0.7, num_shades)]
            color = colors2[i % len(colors2)]
            linestyle = linestyle_tuple[i % len(linestyle_tuple)][1]  # Use a different linestyle for weighted classifiers

        is_soft = (name == "Ensemble_voting_soft")

        is_hard = (name == "Ensemble_voting_hard")

        is_NP = (name == "Neyman_Pearson")

        if is_soft or is_hard:
            # Set distinctive styles for the last line, and standard for others
            linewidth = 3.0 
            linestyle = '--' 
            alpha_value = 0.9 
            plot_zorder = 5 
            marker = 'o'  # Use circle markers for soft/hard voting
            if is_soft:
                color = 'pink'
            else:
                color = 'green'

        if is_NP:
            # Set distinctive styles for the Neyman-Pearson line
            linewidth = 2.5 
            linestyle = ':' 
            alpha_value = 0.9 
            plot_zorder = 4 
            marker = 's'  # Use square markers for Neyman-Pearson
            color = 'red'
        # --- MODIFICATION END ---

        #is_missclassification_risk = (name == "Misclassification_Risk")
        #if is_missclassification_risk:
        #    # Plot the misclassification risk curve with a different style
        #    plt.plot(df_sorted['fpr'], df_sorted['tpr'], color='lime', lw=linewidth,
        #             linestyle='-.', alpha=alpha_value, zorder=plot_zorder + 1,
        #             label=f'{name}', marker=marker)
        #else:
        #    # Plot the line connecting the points for this set
        #    plt.plot(df_sorted['fpr'], df_sorted['tpr'], color=color, lw=linewidth,
        #             linestyle=linestyle, alpha=alpha_value, zorder=plot_zorder,
        #             label=f'{name} (AUC = {roc_auc:.2f})', marker=marker)
        
        plt.plot(df_sorted['fpr'], df_sorted['tpr'], color=color, lw=linewidth,
                     linestyle=linestyle, alpha=alpha_value, zorder=plot_zorder,
                     label=f'{name} (AUC = {roc_auc:.2f})', marker=marker)

        # Plot the individual model points as a scatter plot
        plt.scatter(df_sorted['fpr'], df_sorted['tpr'], c=color, marker=marker, 
                    alpha=0.6, s=80, zorder=plot_zorder + 1) # Place scatter points on top of the line

    # --- Plot the original ROC curve for reference ---
    if results_original_roc and misclassification_risk is not None:
        # Misclassification risk
        #misclassification_risk_orig = prior_prob * (1 - results_original_roc["tpr"]) + (1 - prior_prob) * results_original_roc["fpr"]
        #plt.plot(results_original_roc["fpr"], misclassification_risk_orig, color='cyan', lw=linewidth,
        #         linestyle='-.', alpha=alpha_value, zorder=4,
        #         label=f'Misclassification Risk Original')
        plt.plot(results_original_roc["fpr"], results_original_roc["tpr"], color='blue', lw=2.5,
                 label=f'{results_original_roc["name"]} (AUC = {results_original_roc["auc"]:.2f})')

    # --- Final plot styling ---
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # "No-skill" line
    
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    if misclassification_risk is not None:
        plt.annotate(f'Misclassification risk {results_original_roc["name"]}: {misclassification_risk[0]["risk"]:.2f}', xy=(0.65, 0.2), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        plt.annotate(f'Misclassification risk ensemble: {misclassification_risk[1]["risk"]:.2f}', xy=(0.65, 0.15), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        # Plotting the Optimal Points corresponding to threshold 0.5
        # Point 0: Original Model
        fpr0 = misclassification_risk[0]["fpr"]
        tpr0 = misclassification_risk[0]["tpr"]
        plt.scatter(fpr0, tpr0, color='blue', marker='*', s=250, edgecolors='gold', 
                    linewidths=1.5, zorder=10, label=f'Min Risk Point ({results_original_roc["name"]})')
        
        plt.text(fpr0 + 0.02, tpr0 - 0.02, f"({fpr0:.2f}, {tpr0:.2f})", 
                 fontsize=9, fontweight='bold', color='navy', zorder=11,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        # Point 1: Ensemble Model
        fpr1 = misclassification_risk[1]["fpr"]
        tpr1 = misclassification_risk[1]["tpr"]
        plt.scatter(fpr1, tpr1, color='green', marker='*', s=250, edgecolors='gold', 
                    linewidths=1.5, zorder=10, label='Min Risk Point (Ensemble)')
        
        plt.text(fpr1 + 0.02, tpr1 - 0.02, f"({fpr1:.2f}, {tpr1:.2f})", 
                 fontsize=9, fontweight='bold', color='darkgreen', zorder=11,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))
    
    plt.legend(loc="lower right", fontsize=10)
    full_path_plot = f"Figures/{plot_name}.png"
    plt.savefig(full_path_plot)
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

def save_to_pickle_constrained_roc(constrained_points, filename=''):
    """
    Saves the constrained ROC curve points to a pickle file.

    Args:
        constrained_points (List[Dict]): The constrained ROC curve points.
        filename (str): The name of the file to save the points to.
    """
    with open(filename, 'wb') as f:
        pickle.dump(constrained_points, f)
    print(f"Constrained ROC curve points saved to {filename}")

def load_from_pickle_constrained_roc(filename=''):
    """
    Loads the constrained ROC curve points from a pickle file.

    Args:
        filename (str): The name of the file to load the points from.

    Returns:
        List[Dict]: The constrained ROC curve points.
    """
    with open(filename, 'rb') as f:
        constrained_points = pickle.load(f)
    print(f"Constrained ROC curve points loaded from {filename}")
    return constrained_points

