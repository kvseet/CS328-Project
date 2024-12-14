import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import statsmodels.api as sm
from scipy.stats import norm
import warnings
import seaborn as sns
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
NUM_FOLDS = 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(x):
    return np.log(x / (1 - x))

def datagen(num_samples, seed):
    np.random.seed(seed)
    X = np.random.uniform(low=-1, high=1, size=(num_samples, 3))
    A = np.random.binomial(n=1, p=sigmoid(-1.4 + 0.1 * X[:, 0] + 0.1 * X[:, 1] - 0.1 * X[:, 2]))
    Y = np.random.binomial(n=1, p=(1 - A) * sigmoid(-4.64 + 1/3 * X[:, 0] + 1/3 * X[:, 1] + 1/3 * X[:, 2]))
    df = pd.DataFrame(np.concatenate([X, A.reshape(-1, 1), Y.reshape(-1, 1)], axis=1), columns=['x1', 'x2', 'x3', 'A', 'Y'])
    print("DATA CHECKS")
    pi = np.sum(df['A'] == 1) / num_samples
    print(f"Prevalence of treated: {pi}")
    outcome_rate = np.sum(df[df['A'] == 0]['Y'] == 1) / num_samples
    print(f"Outcome rate in control: {outcome_rate}")
    return df

def create_folds(df):
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=91734135)
    for fold, (_, test_index) in enumerate(kf.split(df), 1):
        df.loc[test_index, 'fold'] = np.int_(fold)
    return df

def fit_superlearner(X_train, Y_train, base_learners, meta_learner):
    fitted_base_learners = {}
    base_predictions_train = pd.DataFrame()
    for name, model in base_learners.items():
        model.fit(X_train, Y_train)
        fitted_base_learners[name] = model
        base_predictions_train[name] = model.predict_proba(X_train)[:, 1]

    meta_learner.fit(base_predictions_train, Y_train)
    return fitted_base_learners, meta_learner

def predict_with_superlearner(test_data, superlearner):
    base_learners, meta_learner = superlearner
    base_predictions_test = np.array([model.predict_proba(test_data)[:, 1] for model in base_learners.values()]).T
    return meta_learner.predict_proba(base_predictions_test)[:, 1]

def fit_outcome_regression(df, fold):
    base_learners = {
        'logistic_regression': LogisticRegression(),
        'random_forest': RandomForestClassifier(n_estimators=10, random_state=1734),
        'svc': SVC(probability=True)
    }
    meta_learner = LogisticRegression()
    filtered_df = df[df['A'] == 0]
    X_train = filtered_df[filtered_df['fold'] != fold].drop(columns=['Y', 'A', 'fold'])
    Y_train = filtered_df[filtered_df['fold'] != fold]['Y']
    return fit_superlearner(X_train, Y_train, base_learners, meta_learner)

def fit_propensity(df, fold):
    base_learners = {
        'logistic_regression': LogisticRegression(),
        'random_forest': RandomForestClassifier(n_estimators=10, random_state=1734),
        'svc': SVC(probability=True)
    }
    meta_learner = LogisticRegression()
    X_train = df[df['fold'] != fold].drop(columns=['Y', 'A', 'fold'])
    Y_train = df[df['fold'] != fold]['A']
    return fit_superlearner(X_train, Y_train, base_learners, meta_learner)

def estimate_outcome_regression(data, outcome_regression):
    return predict_with_superlearner(data, outcome_regression)

def estimate_propensity(data, propensity):
    return predict_with_superlearner(data, propensity)

def estimate_marginal_prob(df, fold):
    return np.sum(df[df['fold'] == fold]['A'] == 1) / len(df[df['fold'] == fold])

def estimate_clever_covariate(X_data, propensity, marginal_prob):
    #print(estimate_propensity(X_data, propensity)[:10])
    return (estimate_propensity(X_data, propensity) / (marginal_prob * (1 - estimate_propensity(X_data, propensity))))

def fit_clever_logistic_regression(df, fold, propensity):
    filtered_df = df[df['fold'] == fold]
    filtered_df = filtered_df[filtered_df['A'] == 0]
    X_data = filtered_df[['x1', 'x2', 'x3']]
    Y_data = filtered_df['Y']
    clever_covariate = estimate_clever_covariate(X_data, propensity, estimate_marginal_prob(df, fold))
    # print(len(clever_covariate))
    # print(len(X_data))
    # print(len(Y_data))
    outcome_regression = fit_outcome_regression(df, fold)
    offset = estimate_outcome_regression(X_data, outcome_regression)
    offset = np.clip(logit(offset), -1e4, 1e4)
    #offset = logit(offset)
    # model = sm.GLM(
    #     Y_data,
    #     clever_covariate,
    #     family=sm.families.Binomial(),
    #     offset=offset
    # ).fit()
    model = smf.glm(
        'Y ~ clever_covariate - 1',
        data=pd.DataFrame({'Y': Y_data, 'clever_covariate': clever_covariate}),
        family=sm.families.Binomial(),
        offset=offset
    ).fit()
    epsilon_v_star = model.params[0]
    return epsilon_v_star

#pass in propensities as a list for each fold
def fit_clever_logistic_regression_pooled(df, propensities, num_folds=NUM_FOLDS):
    clever_covariates = np.array([])
    offset = np.array([])
    for fold in range(1, num_folds + 1):
        filtered_df = df[df['fold'] == fold]
        filtered_df = filtered_df[filtered_df['A'] == 0]
        X_data = filtered_df[['x1', 'x2', 'x3']]
        marginal_prob_fold = estimate_marginal_prob(df, fold)
        clever_covariates = np.append(clever_covariates, estimate_clever_covariate(X_data, propensities[fold - 1], marginal_prob_fold))
        outcome_regression_fold = fit_outcome_regression(df, fold)
        offset = np.append(offset, estimate_outcome_regression(X_data, outcome_regression_fold))
    clever_covariates = np.array(clever_covariates)
    offset = np.array(offset)
    offset = np.clip(logit(offset), -1e4, 1e4)
    #offset = logit(offset)
    Y_data = df[df['A'] == 0]['Y']
    model = smf.glm(
        'Y ~ clever_covariate - 1',
        data=pd.DataFrame({'Y': Y_data, 'clever_covariate': clever_covariates}),
        family=sm.families.Binomial(),
        offset=offset
    ).fit()
    epsilon_v_star = model.params[0]
    return epsilon_v_star

def clever_targeted_outcome_regression(epsilon_v_star, outcome_regression, clever_covariate):
    return sigmoid(np.clip(logit(outcome_regression), -1e4, 1e4) + epsilon_v_star * clever_covariate)

def clever_plug_in_estimator(targeted_outcome_regression, df, fold):
    filtered_df = df[df['fold'] == fold]
    print("PLUG IN VALS")
    print(np.sum((filtered_df['A'] == 1) * targeted_outcome_regression))
    print(np.sum(filtered_df['A'] == 1))
    return np.sum((filtered_df['A'] == 1) * targeted_outcome_regression) / np.sum(filtered_df['A'] == 1)

def clever_cross_validated_plug_in_estimator(df, num_folds=NUM_FOLDS):
    plug_in_estimators = []
    sum_of_squared_bias = 0
    propensity_values = []
    targeted_outcome_regression_values = []
    marginal_prob_values = []
    for fold in range(1, num_folds + 1):
        outcome_regression = fit_outcome_regression(df, fold)
        propensity = fit_propensity(df, fold)
        marginal_prob = estimate_marginal_prob(df, fold)
        epsilon_v_star = fit_clever_logistic_regression(df, fold, propensity)
        print(epsilon_v_star)
        outcome_regression_values = estimate_outcome_regression(df[df['fold'] == fold][['x1', 'x2', 'x3']], outcome_regression)
        clever_covariate_values = estimate_clever_covariate(df[df['fold'] == fold][['x1', 'x2', 'x3']], propensity, marginal_prob)
        targeted_outcome_regression = clever_targeted_outcome_regression(epsilon_v_star, outcome_regression_values, clever_covariate_values)
        fold_psi = clever_plug_in_estimator(targeted_outcome_regression, df, fold)
        plug_in_estimators.append(fold_psi)
        fold_values = df[df['fold'] == fold]
        propensity_values.append(estimate_propensity(fold_values[['x1', 'x2', 'x3']], propensity))
        targeted_outcome_regression_values.append(targeted_outcome_regression)
        marginal_prob_values.append(marginal_prob)
    #print(plug_in_estimators)
    psi = np.mean(plug_in_estimators) 
    # print(propensity_values)
    # print(targeted_outcome_regression_values)
    # print(marginal_prob_values)
    for fold in range(1, num_folds + 1):
        sum_of_squared_bias += compute_sum_of_squared_bias(targeted_outcome_regression_values[fold - 1], propensity_values[fold - 1], marginal_prob_values[fold - 1], df, fold, psi)
    return psi, (1 / np.sqrt(len(df))) * (sum_of_squared_bias / len(df)) ** (0.5)

def pooled_outcome_regression_values(df, outcome_regressions, num_folds=NUM_FOLDS):
    pooled_outcome_regression_values = []
    for i in range(len(df)):
        fold = df.iloc[i]['fold']
        fold = int(fold)
        pooled_outcome_regression_values.append(estimate_outcome_regression(np.array(df.iloc[i][['x1', 'x2', 'x3']]).reshape(1, -1), outcome_regressions[fold - 1]))
    return np.array(pooled_outcome_regression_values)

def pooled_clever_covariates(df, propensities, marginal_prob_values, num_folds=NUM_FOLDS):
    pooled_clever_covariates = []
    for i in range(len(df)):
        fold = df.iloc[i]['fold']
        fold = int(fold)
        pooled_clever_covariates.append(estimate_clever_covariate(np.array(df.iloc[i][['x1', 'x2', 'x3']]).reshape(1, -1), propensities[fold - 1], marginal_prob_values[fold - 1]))
    return np.array(pooled_clever_covariates)

def clever_pooled_plug_in_estimator(df, num_folds=NUM_FOLDS):
    marginal_prob_values = []
    outcome_regressions = []
    propensities = []
    for fold in range(1, num_folds + 1):
        outcome_regression = fit_outcome_regression(df, fold)
        outcome_regressions.append(outcome_regression)
        propensity = fit_propensity(df, fold)
        propensities.append(propensity)
        marginal_prob = estimate_marginal_prob(df, fold)
        marginal_prob_values.append(marginal_prob)
    outcome_regression_values = pooled_outcome_regression_values(df, outcome_regressions, num_folds)
    clever_covariates = pooled_clever_covariates(df, propensities, marginal_prob_values, num_folds)
    epsilon_v_star = fit_clever_logistic_regression_pooled(df, propensities, num_folds)
    #print(epsilon_v_star)
    targeted_outcome_regression = clever_targeted_outcome_regression(epsilon_v_star, outcome_regression_values, clever_covariates)
    psi = np.sum((df['A'] == 1) * targeted_outcome_regression.squeeze()) / np.sum(df['A'] == 1)
    return psi

def clever_estimate_theta(df, psi):
    #print(np.sum((df['A'] == 1) * df['Y']))
    #print(np.sum(df['A'] == 1))
    return (np.sum((df['A'] == 1) * df['Y'])) / (np.sum(df['A'] == 1)) - psi

def compute_sum_of_squared_bias(targeted_outcome_regression, propensity, marginal_prob, df, fold, psi):
    filtered_df = df[df['fold'] == fold]
    num = np.where(filtered_df['A'] == 0, 1, 0) * propensity
    denom = marginal_prob * (1 - propensity)
    residuals = filtered_df['Y'] - targeted_outcome_regression
    first_term = num / denom * residuals
    second_term = np.where(filtered_df['A'] == 1, 1, 0) / marginal_prob * (targeted_outcome_regression - psi)
    # print(first_term)
    # print(second_term)
    return np.sum((first_term + second_term) ** 2)

def weights(df, fold, propensity, marginal_prob):
    filtered_df = df[df['fold'] == fold]
    propensity_values = estimate_propensity(filtered_df[['x1', 'x2', 'x3']], propensity)
    num = np.where(filtered_df['A'] == 0, 1, 0) * propensity_values
    denom = marginal_prob * (1 - propensity_values)
    return num / denom

def intercept_only_weight_regression(df, fold, weights_values, outcome_regression, marginal_prob):
    filtered_df = df[df['fold'] == fold]
    X_data = np.ones(len(filtered_df))
    Y_data = np.array(filtered_df['Y'])
    offset = estimate_outcome_regression(filtered_df[['x1', 'x2', 'x3']], outcome_regression)
    offset = np.clip(logit(offset), -1e4, 1e4)
    model = smf.glm(
        'Y ~ 1',
        data=pd.DataFrame({'Y': Y_data}),
        family=sm.families.Binomial(),
        offset=offset,
        freq_weights=weights_values
    ).fit()
    epsilon_v_star = model.params[0]
    return epsilon_v_star

def intercept_only_weight_regression_pooled(df, weights_values, outcome_regression_values):
    X_data = np.ones(len(df))
    Y_data = np.array(df['Y'])
    model = smf.glm(
        'Y ~ 1',
        data=pd.DataFrame({'Y': Y_data}),
        family=sm.families.Binomial(),
        offset=np.clip(logit(outcome_regression_values), -1e4, 1e4),
        freq_weights=weights_values
    ).fit()
    epsilon_v_star = model.params[0]
    return epsilon_v_star

def weighted_cross_validated_plug_in_estimator(df, num_folds=NUM_FOLDS):
    plug_in_estimators = []
    sum_of_squared_bias = 0
    propensity_values = []
    targeted_outcome_regression_values = []
    marginal_prob_values = []
    for fold in range(1, num_folds + 1):
        outcome_regression = fit_outcome_regression(df, fold)
        propensity = fit_propensity(df, fold)
        marginal_prob = estimate_marginal_prob(df, fold)
        weights_values = weights(df, fold, propensity, marginal_prob)
        epsilon_v_star = intercept_only_weight_regression(df, fold, weights_values, outcome_regression, marginal_prob)
        print(epsilon_v_star)
        outcome_regression_values = estimate_outcome_regression(df[df['fold'] == fold][['x1', 'x2', 'x3']], outcome_regression)
        targeted_outcome_regression = sigmoid(np.clip(logit(outcome_regression_values), -1e4, 1e4) + epsilon_v_star)
        fold_psi = clever_plug_in_estimator(targeted_outcome_regression, df, fold)
        plug_in_estimators.append(fold_psi)
        fold_values = df[df['fold'] == fold]
        propensity_values.append(estimate_propensity(fold_values[['x1', 'x2', 'x3']], propensity))
        targeted_outcome_regression_values.append(targeted_outcome_regression)
        marginal_prob_values.append(marginal_prob)
    psi = np.mean(plug_in_estimators)
    # print(propensity_values)
    # print(targeted_outcome_regression_values)
    # print(marginal_prob_values)
    for fold in range(1, num_folds + 1):
        sum_of_squared_bias += compute_sum_of_squared_bias(targeted_outcome_regression_values[fold - 1], propensity_values[fold - 1], marginal_prob_values[fold - 1], df, fold, psi)
    return psi, (1 / np.sqrt(len(df))) * (sum_of_squared_bias / len(df)) ** (0.5)

def pooled_weights(df, propensities, marginal_prob_values, num_folds=NUM_FOLDS):
    pooled_weights = []
    for i in range(len(df)):
        fold = df.iloc[i]['fold']
        fold = int(fold)
        propensity_value = estimate_propensity(np.array(df.iloc[i][['x1', 'x2', 'x3']]).reshape(1, -1), propensities[fold - 1])
        pooled_weights.append((df.loc[i, 'A'] == 0) * propensity_value / (marginal_prob_values[fold - 1] * (1 - propensity_value)))
    return np.array(pooled_weights)

def pooled_weighted_plug_in_estimator(df, num_folds=NUM_FOLDS):
    marginal_prob_values = []
    outcome_regressions = []
    propensities = []
    for fold in range(1, num_folds + 1):
        outcome_regression = fit_outcome_regression(df, fold)
        outcome_regressions.append(outcome_regression)
        propensity = fit_propensity(df, fold)
        propensities.append(propensity)
        marginal_prob = estimate_marginal_prob(df, fold)
        marginal_prob_values.append(marginal_prob)
    outcome_regression_values = pooled_outcome_regression_values(df, outcome_regressions, num_folds)
    weights_values = pooled_weights(df, propensities, marginal_prob_values, num_folds)
    #print(outcome_regression_values.shape)
    epsilon_v_star = intercept_only_weight_regression_pooled(df, weights_values.squeeze(), outcome_regression_values.squeeze())
    print(epsilon_v_star)
    targeted_outcome_regression = sigmoid(np.clip(logit(outcome_regression_values), -1e4, 1e4) + epsilon_v_star)
    psi = np.sum((df['A'] == 1) * targeted_outcome_regression.squeeze()) / np.sum(df['A'] == 1)
    return psi

def compute_dml_estimator(df, fold, outcome_regression, propensity, marginal_prob):
    filtered_df = df[df['fold'] == fold]
    first_term = np.sum((filtered_df['A'] == 1) * outcome_regression) / np.sum(filtered_df['A'] == 1)
    residuals = filtered_df['Y'] - outcome_regression
    second_term = np.sum((filtered_df['A'] == 0) * propensity / (marginal_prob * (1 - propensity)) * residuals)
    print("HERE")
    print(first_term / len(filtered_df))
    print(second_term / len(filtered_df))
    return 1/len(filtered_df) * (first_term + second_term)

def dml_estimator(df, num_folds=NUM_FOLDS):
    plug_in_estimators = []
    outcome_regression_values = []
    propensity_values = []
    marginal_prob_values = []
    sum_of_squared_bias = 0
    for fold in range(1, num_folds + 1):
        filtered_df = df[df['fold'] == fold]
        X_data = filtered_df[['x1', 'x2', 'x3']]
        outcome_regression = fit_outcome_regression(df, fold)
        outcome_regression_values.append(estimate_outcome_regression(X_data, outcome_regression))
        propensity = fit_propensity(df, fold)
        propensity_values.append(estimate_propensity(X_data, propensity))
        marginal_prob_values.append(estimate_marginal_prob(df, fold))
        fold_psi = compute_dml_estimator(df, fold, outcome_regression_values[-1], propensity_values[-1], marginal_prob_values[-1])
        plug_in_estimators.append(fold_psi)
    psi = np.mean(plug_in_estimators)
    for fold in range(1, num_folds + 1):
        sum_of_squared_bias += compute_sum_of_squared_bias(outcome_regression_values[fold - 1], propensity_values[fold - 1], marginal_prob_values[fold - 1], df, fold, psi)
    return psi, ((1 / np.sqrt(len(df))) * (sum_of_squared_bias / len(df)) ** (0.5))

def dml_estimator_clipped(df, num_folds=NUM_FOLDS):
    plug_in_estimators = []
    outcome_regression_values = []
    propensity_values = []
    marginal_prob_values = []
    sum_of_squared_bias = 0
    for fold in range(1, num_folds + 1):
        filtered_df = df[df['fold'] == fold]
        X_data = filtered_df[['x1', 'x2', 'x3']]
        outcome_regression = fit_outcome_regression(df, fold)
        outcome_regression_values.append(estimate_outcome_regression(X_data, outcome_regression))
        propensity = fit_propensity(df, fold)
        propensity_values.append(estimate_propensity(X_data, propensity))
        marginal_prob_values.append(estimate_marginal_prob(df, fold))
        fold_psi = compute_dml_estimator(df, fold, outcome_regression_values[-1], propensity_values[-1], marginal_prob_values[-1])
        plug_in_estimators.append(fold_psi)
    psi = np.mean(plug_in_estimators)
    psi = np.clip(psi, 0, 1)
    for fold in range(1, num_folds + 1):
        sum_of_squared_bias += compute_sum_of_squared_bias(outcome_regression_values[fold - 1], propensity_values[fold - 1], marginal_prob_values[fold - 1], df, fold, psi)
    return psi, (1 / np.sqrt(len(df))) * (sum_of_squared_bias / len(df)) ** (0.5)

def compute_wald_ci(psi, standard_error, alpha=0.05):
    return psi - norm.ppf(1 - alpha / 2) * standard_error, psi + norm.ppf(1 - alpha / 2) * standard_error

results = {}
def main():
    results['tmle_c'] = []
    results['tmle_w'] = []
    results['tmle_cp'] = []
    results['tmle_wp'] = []
    results['dml'] = []
    results['dml_clipped'] = []
    for seed in range(1, 101):
        print("SEED")
        print(seed)
        df = datagen(2000, seed)
        print(df.head())
        df = create_folds(df)
        print(df.head())
        tmle_c, tmle_c_standard_error = clever_cross_validated_plug_in_estimator(df)
        print(tmle_c)
        print(clever_estimate_theta(df, tmle_c))
        print(compute_wald_ci(tmle_c, tmle_c_standard_error))
        if clever_estimate_theta(df, tmle_c) > -0.05:
            results['tmle_c'].append(clever_estimate_theta(df, tmle_c))
        tmle_w, tmle_w_standard_error = weighted_cross_validated_plug_in_estimator(df)
        print(tmle_w)
        print(clever_estimate_theta(df, tmle_w))
        print(compute_wald_ci(tmle_w, tmle_w_standard_error))
        if clever_estimate_theta(df, tmle_w) > -0.05:
            results['tmle_w'].append(clever_estimate_theta(df, tmle_w))
        tmle_cp = clever_pooled_plug_in_estimator(df)
        print(tmle_cp)
        print(clever_estimate_theta(df, tmle_cp))
        if clever_estimate_theta(df, tmle_cp) > -0.05:
            results['tmle_cp'].append(clever_estimate_theta(df, tmle_cp))
        tmle_wp = pooled_weighted_plug_in_estimator(df)
        print(tmle_wp)
        print(clever_estimate_theta(df, tmle_wp))
        if clever_estimate_theta(df, tmle_wp) > -0.05:
            results['tmle_wp'].append(clever_estimate_theta(df, tmle_wp))
        dml, dml_standard_error = dml_estimator(df)
        print(dml)
        print(clever_estimate_theta(df, dml))
        print(compute_wald_ci(dml, dml_standard_error))
        if clever_estimate_theta(df, dml) > -0.05:
            results['dml'].append(clever_estimate_theta(df, dml))
        dml_clipped, dml_clipped_standard_error = dml_estimator_clipped(df)
        print(dml_clipped)
        print(clever_estimate_theta(df, dml_clipped))
        print(compute_wald_ci(dml_clipped, dml_clipped_standard_error))
        if clever_estimate_theta(df, dml_clipped) > -0.05:
            results['dml_clipped'].append(clever_estimate_theta(df, dml_clipped))
    data = pd.DataFrame({key: pd.Series(value) for key, value in results.items()})
    data_melted = data.melt(var_name="method", value_name="estimate")
    print(data_melted)

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data_melted, x="method", y="estimate", inner=None, color="lightgray", linewidth=2.0, cut=0)

    # Add means and medians
    methods = data_melted['method'].unique()
    for method in methods:
        estimates = results[method]
        mean = np.mean(estimates)
        median = np.median(estimates)
        x = list(methods).index(method)
        
        plt.scatter(x, mean, color='gray', label="Mean" if method == methods[0] else "", s=50)
        plt.scatter(x, median, color='black', marker='^', label="Median" if method == methods[0] else "", s=50)

    # Add labels and style
    plt.axhline(-0.0101, color="gray", linestyle="--", linewidth=1)  # Truth line
    plt.legend(loc="upper left")
    plt.title("Violin Plot of ATT Estimates", fontsize=14)
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Estimate of ATT θ", fontsize=12)
    plt.tight_layout()
    plt.savefig("n=2000.png")

if __name__ == "__main__":
    main()