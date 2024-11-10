import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy.stats import ttest_ind, mannwhitneyu, kruskal, chi2_contingency, normaltest
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

def show_box_boxwithout_hist(x, columns, df_out, kde=False, path_to_save=None):
    df = df_out.copy()
    
    height = len(columns) * 4

    # Define colors for each group
    palette = {0: "#FFC107", 1: "#004D40"}

    fig, axs = plt.subplots(ncols=3, nrows=len(columns), figsize=(15, height))
    for idx, col in enumerate(columns):
        ax = sns.boxplot(x=x, y=col, data=df, showfliers=True, ax=axs[idx][0], hue=x, legend=False, palette=palette)
        ax.set_xlabel('') # Disable x-axis label
        ax.set_ylabel(col, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)     
        ax.tick_params(axis='x', labelsize=12)
        ax = sns.boxplot(x=x, y=col, data=df, showfliers=False, ax=axs[idx][1], hue=x, legend=False, palette=palette)
        ax.set_xlabel('') # Disable x-axis label
        ax.set_ylabel(col, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)

        try:
            ax = sns.histplot(x=col, data=df, hue=x, kde=kde, ax=axs[idx][2], palette=palette)
            ax.set_xlabel('') # Disable x-axis label
            ax.set_ylabel("Count", fontsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.tick_params(axis='x', labelsize=12)
        except:
            print("Error - col {}".format(col))

    # Adjust layout
    plt.tight_layout()

    if path_to_save is not None:
        # Save the plot to a file
        plt.savefig(path_to_save, bbox_inches='tight')

    plt.show()

def detect_categorical_columns(df, threshold=3):
    """
    Detects categorical columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    threshold (int): The maximum number of unique values for a column to be considered categorical (default is 10).
    
    Returns:
    categorical_cols (list): List of column names that are considered categorical.
    continuous_cols (list): List of column names that are considered continuous.
    """
    categorical_cols = []
    continuous_cols = []
    
    for column in df.columns:
        unique_values = df[column].nunique()
        
        if pd.api.types.is_numeric_dtype(df[column]):
            # If column is numeric but has few unique values (e.g., binary or limited categories), treat as categorical
            if unique_values <= threshold:
                categorical_cols.append(column)
            else:
                continuous_cols.append(column)
        else:
            # Non-numeric columns are categorical
            categorical_cols.append(column)
    
    return categorical_cols, continuous_cols

def calculate_advanced_descriptive_stats(target, continuous_features, categorical_features, dataset, path):
    # Select the target and feature columns
    data = dataset[[target] + continuous_features + categorical_features]
    
    # Create an Excel writer object
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        # Descriptive statistics
        descriptive_stats = data.describe()
        descriptive_stats.to_excel(writer, sheet_name='Descriptive Stats')
        
        # Skewness
        skewness = data[continuous_features].skew()
        skewness.to_frame(name='Skewness').to_excel(writer, sheet_name='Skewness')
        
        # Kurtosis
        kurtosis = data[continuous_features].kurt()
        kurtosis.to_frame(name='Kurtosis').to_excel(writer, sheet_name='Kurtosis')
        
        # Correlation matrix for continuous features
        correlation_matrix = data[continuous_features].corr()
        correlation_matrix.to_excel(writer, sheet_name='Correlation Matrix')
        
        # Statistical tests (t-test, Mann-Whitney U, Kruskal-Wallis H, Chi-squared)
        stats_results = []
        
        # Continuous features: T-test, Mann-Whitney U test, or Kruskal-Wallis H test
        for feature in continuous_features:
            # Check if target is binary (use t-test or U test)
            if dataset[target].nunique() == 2:
                # Perform normality test
                stat, p_normality = normaltest(data[feature])
                
                if p_normality > 0.05:  # Normally distributed
                    try:
                        # Perform t-test
                        t_stat, p_value = ttest_ind(data[data[target] == 0][feature], data[data[target] == 1][feature])
                        
                        # Simplified degrees of freedom for two-sample t-test
                        n1 = len(data[data[target] == 0])
                        n2 = len(data[data[target] == 1])
                        degrees_of_freedom = n1 + n2 - 2
                        
                        stats_results.append((feature, 'Two-sample t-test', t_stat, p_value, degrees_of_freedom))
                    except Exception as e:
                        stats_results.append((feature, 'Two-sample t-test', 'Error', str(e), None))
                else:  # Not normally distributed
                    try:
                        # Perform Mann-Whitney U test
                        u_stat, p_value = mannwhitneyu(data[data[target] == 0][feature], data[data[target] == 1][feature], alternative='two-sided')
                        
                        # Calculate the z-value for Mann-Whitney U test
                        n1 = len(data[data[target] == 0])
                        n2 = len(data[data[target] == 1])
                        mean_u = n1 * n2 / 2
                        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                        z_value = (u_stat - mean_u) / std_u
                        
                        stats_results.append((feature, 'Mann-Whitney U test', u_stat, p_value, z_value))
                    except Exception as e:
                        stats_results.append((feature, 'Mann-Whitney U test', 'Error', str(e), None))
            else:
                # More than two groups: Use Kruskal-Wallis H test
                try:
                    h_stat, p_value = kruskal(*[data[data[target] == group][feature] for group in data[target].unique()])
                    stats_results.append((feature, 'Kruskal-Wallis H test', h_stat, p_value, None))
                except Exception as e:
                    stats_results.append((feature, 'Kruskal-Wallis H test', 'Error', str(e), None))
        
        # Categorical features: Chi-squared test
        for feature in categorical_features:
            try:
                contingency_table = pd.crosstab(dataset[target], dataset[feature])
                chi2_stat, p_value, _, expected = chi2_contingency(contingency_table)
                
                # Calculate degrees of freedom for chi-squared test
                degrees_of_freedom = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)
                
                stats_results.append((feature, 'Chi-squared test', chi2_stat, p_value, degrees_of_freedom))
            except Exception as e:
                stats_results.append((feature, 'Chi-squared test', 'Error', str(e), None))
        
        # Convert results to a DataFrame and save to Excel
        stats_df = pd.DataFrame(stats_results, columns=['Feature', 'Test Type', 'Test Statistic', 'P-Value', 'Additional Info (z-value or df)'])
        stats_df.to_excel(writer, sheet_name='Statistical Tests')
        
        # The writer will automatically save and close the file when using 'with' context

    print(f"Descriptive statistics, correlation matrix, and statistical tests saved to {path}")

def calculate_descriptive_stats(target, features, dataset, path):
    # Select the target and feature columns
    data = dataset[[target] + features]
    
    # Create an Excel writer object
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        # Descriptive statistics
        descriptive_stats = data.describe()
        descriptive_stats.to_excel(writer, sheet_name='Descriptive Stats')
        
        # Skewness
        skewness = data.skew()
        skewness.to_frame(name='Skewness').to_excel(writer, sheet_name='Skewness')
        
        # Kurtosis
        kurtosis = data.kurt()
        kurtosis.to_frame(name='Kurtosis').to_excel(writer, sheet_name='Kurtosis')
        
        # Correlation matrix
        correlation_matrix = data.corr()
        correlation_matrix.to_excel(writer, sheet_name='Correlation Matrix')
        
        # The writer will automatically save and close the file when using 'with' context

    print(f"Descriptive statistics and correlation matrix saved to {path}")

def cohen_d(x1, x2):
    nx1 = len(x1)
    nx2 = len(x2)
    s = np.sqrt(((nx1-1) * np.std(x1, ddof=1)**2 + (nx2-1) * np.std(x2, ddof=1)**2) / (nx1 + nx2 - 2))
    return (np.abs(np.mean(x1) - np.mean(x2))) / s

def test_feature(df, feature_name, results, logging, ignore_power=False):
    feature_i = df.loc[(df['indicator_fg'] == 1) & (df[feature_name].notnull()), feature_name]  

    feature_ni = df.loc[(df['indicator_fg'] == 0) & (df[feature_name].notnull()), feature_name]
    
    alpha = 0.05
    t_test = True
    selected = True
    
    t_u_p = [feature_name, -1, -1, -1, -1, -1, -1]
    
    logging.info("----------Shapiro test----------")
    shapiro_test_i = stats.shapiro(feature_i)
    shapiro_test_ni = stats.shapiro(feature_ni)
    
    if shapiro_test_i.pvalue < alpha and shapiro_test_ni.pvalue < alpha:
        logging.info("The null hypothesis that the data was drawn from a normal distribution can be rejected")
        t_test = False
    else:
        logging.info("The null hypothesis that the data was drawn from a normal distribution cannot be rejected")
    
    logging.info("----------Levene test----------")
    levene_test = stats.levene(feature_i, feature_ni)
    
    if levene_test.pvalue < alpha:
        logging.info("The null hypothesis that all input samples are from populations with equal variances can be rejected")
        t_test = False
    else:
        logging.info("The null hypothesis that all input samples are from populations with equal variances cannot be rejected")
    
    if t_test:
        logging.info("----------T test----------")
        t_test = stats.ttest_ind(feature_i, feature_ni)
        logging.info(t_test)
        t_u_p[1] = t_test.statistic
        t_u_p[2] = t_test.pvalue
        if t_test.pvalue < alpha:
            logging.info("The null hypothesis that 2 independent samples have identical average (expected) values can be rejected")
        else:
            logging.info("The null hypothesis that 2 independent samples have identical average (expected) values cannot be rejected")
            selected = False
    else:
        logging.info("----------U test----------")
        u_test = stats.mannwhitneyu(feature_i, feature_ni)
        logging.info(u_test)
        t_u_p[3] = u_test.statistic
        t_u_p[4] = u_test.pvalue
        if u_test.pvalue < alpha:
            logging.info("The null hypothesis that the distribution underlying sample x is the same as the distribution underlying sample y can be rejected")
        else:
            logging.info("The null hypothesis that the distribution underlying sample x is the same as the distribution underlying sample y cannot be rejected")
            selected = False

    logging.info("----------Power----------")
    feature_c_d = cohen_d(feature_i, feature_ni)
    power_feature = sm_stats.power.tt_ind_solve_power(feature_c_d, len(feature_i), 0.05, None, 1)
    t_u_p[5] = power_feature
    if power_feature > 0.7:
        logging.info(f"Sufficient power {power_feature}")
    else:
        logging.info(f"Not sufficient power {power_feature}")
        if not ignore_power:
            selected = False

    logging.info(f"{feature_name}: {selected}")
    # print(feature_name, selected)
    t_u_p[6] = selected
    results.append(t_u_p)
    return selected


# The following function is taken from my project developed on the subject Intelligent Data Analysis 2021/2022.
def report_generator(pred_train, pred_test, y_train, y_test, driver_silent, zero_division, logging):
    if not driver_silent:
        print("Predicting for train dataset:")
        print(classification_report(y_train, pred_train, zero_division=zero_division))

        print("Predicting for test dataset:")
        print(classification_report(y_test, pred_test, zero_division=zero_division))
    
    report_train = classification_report(y_train, pred_train, output_dict=True, zero_division=zero_division)
    report_test = classification_report(y_test, pred_test, output_dict=True, zero_division=zero_division)

    logging.info("Predicting for train dataset:")
    logging.info(classification_report(y_train, pred_train, zero_division=zero_division))
    logging.info("Detailed classification report:")
    logging.info(report_train)

    logging.info("Predicting for test dataset:")
    logging.info(classification_report(y_test, pred_test, zero_division=zero_division))
    logging.info("Detailed classification report:")
    logging.info(report_test)
    
    return report_train, report_test

def model_training_old(clf, X_train, X_test, y_train, y_test, logging, save_path, grid, random_state, n_iter=100, cv=5, scoring='balanced_accuracy', verbose=0, driver_silent=True, zero_division='warn'):
    logging.info(f"Training model {clf}")

    clf.fit(X_train, y_train)

    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    # Save the model
    if save_path is not None:
        logging.info(f"Saving model to {save_path}")
        joblib.dump(clf, save_path)

    params = {"no": "params"}

    return clf, params, *report_generator(pred_train, pred_test, y_train, y_test, driver_silent, zero_division, logging)

def model_training(clf, X_train, X_test, y_train, y_test, logging, save_path, grid, random_state, n_iter=100, cv=5, scoring='balanced_accuracy', verbose=0, driver_silent=True, zero_division='warn'):
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    logging.info(f"Starting hyperparameter tuning for {clf}")

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=grid, n_iter=n_iter, cv=cv, scoring=scoring, random_state=random_state, verbose=verbose)
    random_search.fit(X_train, y_train)

    # Best estimator and parameters after the random search
    best_clf = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Logging and printing the best parameters
    logging.info(f"Best parameters found: {best_params}")
    print(f"Best parameters: {best_params}")  # Print statement for best parameters

    pred_train = best_clf.predict(X_train)
    pred_test = best_clf.predict(X_test)

    # Save the model
    if save_path is not None:
        logging.info(f"Saving model to {save_path}")
        joblib.dump(best_clf, save_path)

    return best_clf, best_params, *report_generator(pred_train, pred_test, y_train, y_test, driver_silent, zero_division, logging)


def _add_to_global_report(global_report, report, algo, d_set, best_params):
    dictionary = report.copy()
    dictionary['algorithm'] = algo
    dictionary['set'] = d_set
    dictionary['best_params'] = str(best_params)
    pd_temp = pd.DataFrame({ key: dictionary[key] for key in ['algorithm', 'set', 'accuracy', '0', '1', 'macro avg', 'weighted avg', 'best_params'] })
    
    if global_report is None:
        global_report = pd_temp
    else:
        global_report = pd.concat([global_report, pd_temp], axis=0)
    
    return global_report

def add_to_global_report(global_report, report_train, report_test, algo, best_params):
    global_report = _add_to_global_report(global_report, report_train, algo, 'train', best_params)
    global_report = _add_to_global_report(global_report, report_test, algo, 'test', best_params)
    return global_report

def calculate_shap(clf, X_train, X_test, tree=False, pos_class=False):
    if tree:
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.Explainer(clf, X_train)

    shap_values = explainer(X_test)

    if pos_class:
        shap_values = shap_values[..., 1]

    shap.plots.beeswarm(shap_values, max_display=10, order=shap.Explanation.abs.mean(0))
    shap.plots.beeswarm(shap_values, max_display=100, order=shap.Explanation.abs.mean(0))