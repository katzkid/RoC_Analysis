import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.base import BaseEstimator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def predict_ensemble_and_evaluate(list_folds_best_models, test_loader, target_fpr):
    """
    Performs soft and hard voting for PyTorch or Scikit-learn models,
    using a PyTorch DataLoader as input.

    Args:
        list_folds_best_models (List[List[Dict]]): Contains model snapshots.
        test_loader (DataLoader): DataLoader for the test dataset.
        target_fpr (float): The target FPR to aim for.

    Returns:
        dict: A dictionary with 'tpr' and 'fpr' for both methods, or None on error.
    """
    # --- Part 1: Extract Full Dataset from Loader for Scikit-Learn Compatibility ---
    print("Extracting full dataset from DataLoader for scikit-learn compatibility...")
    X_test_list, y_test_list = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Flatten image data for universal compatibility with sklearn models
            # For tabular data, this reshape does nothing harmful
            X_test_list.append(inputs.cpu().numpy().reshape(len(inputs), -1))
            y_test_list.append(labels.cpu().numpy())
            
    X_test = np.vstack(X_test_list)
    true_labels = np.concatenate(y_test_list).flatten()
    print(f"-> Extracted {len(true_labels)} samples.\n")

    # --- Part 2: Select Best Models ---
    best_models_info = []
    for i, fold_run_data in enumerate(list_folds_best_models):
        best_snapshot = min(
            (s for s in fold_run_data if "fpr" in s),
            key=lambda s: abs(s["fpr"] - target_fpr),
            default=None
        )
        if best_snapshot:
            best_models_info.append(best_snapshot)
        else:
            print(f"Warning: Could not find a suitable model in Fold {i+1}. Skipping.")

    if not best_models_info:
        print("Error: No models were found. Cannot proceed.")
        return None

    # --- Part 3: Get Probabilities from Each Model (Unified Logic) ---
    all_fold_probas = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for snapshot in best_models_info:
        model = snapshot['model']
        fold_probas = None

        # 🧠 PyTorch Model Logic (uses the efficient DataLoader)
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

        # 🤖 Scikit-learn Model Logic (uses the extracted NumPy array)
        elif isinstance(model, BaseEstimator) and hasattr(model, 'predict_proba'):
            # Uses the X_test array we created in Part 1
            probas = model.predict_proba(X_test)
            fold_probas = probas[:, 1]
        
        else:
            print(f"Warning: Skipping model of unsupported type: {type(model)}")
            continue
            
        all_fold_probas.append(fold_probas)
    
    # --- Part 4: Calculate Results for Both Voting Methods ---
    results = {}

    if not all_fold_probas or np.array(all_fold_probas).size == 0:
        print("Error: No model probabilities were generated. Cannot calculate metrics.")
        return None

    # -- 🍦 Soft Voting --
    ensemble_probas = np.mean(np.array(all_fold_probas), axis=0)
    fpr_sv, tpr_sv, _ = roc_curve(true_labels, ensemble_probas)
    idx_sv = np.argmin(np.abs(fpr_sv - target_fpr))
    results['soft_voting'] = {'tpr': tpr_sv[idx_sv], 'fpr': fpr_sv[idx_sv]}
    
    # -- 🗳️ Hard Voting --
    all_fold_hard_preds = []
    for i, probas in enumerate(all_fold_probas):
        model_threshold = best_models_info[i]['threshold']
        
        # FIX: Convert tensor to float before comparison
        if isinstance(model_threshold, torch.Tensor):
            model_threshold = model_threshold.item()
            
        hard_preds = (probas >= model_threshold).astype(int)
        all_fold_hard_preds.append(hard_preds)
    
    sum_of_votes = np.sum(np.array(all_fold_hard_preds), axis=0)
    num_models = len(all_fold_hard_preds)
    final_hard_preds = (sum_of_votes > num_models / 2).astype(int)
    
    # Use ravel() to handle multi-class confusion matrices if they arise
    tn, fp, fn, tp = confusion_matrix(true_labels, final_hard_preds).ravel()
    hard_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    hard_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    results['hard_voting'] = {'tpr': hard_tpr, 'fpr': hard_fpr}

    # --- Part 5: Print Summary and Return ---
    print("\n--- Ensemble Results ---")
    print(f"Target FPR: {target_fpr:.4f}")
    print(f"Soft Voting -> Achieved [TPR: {results['soft_voting']['tpr']:.4f}, FPR: {results['soft_voting']['fpr']:.4f}]")
    print(f"Hard Voting -> Resulted in [TPR: {results['hard_voting']['tpr']:.4f}, FPR: {results['hard_voting']['fpr']:.4f}]")
    
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