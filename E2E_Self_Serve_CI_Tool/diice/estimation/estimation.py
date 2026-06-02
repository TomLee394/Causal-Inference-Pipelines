import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import econml
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

class PropensityBasedEstimation():
    
    """
    Main class for propensity methods (Matching, Inverse Propensity Weighting). Instances of this class will take the input data (i.e. adjustment set from the identification module) perform sample splitting (only supports 2 splits for now) and cross fit propensity models to each split. Propensity method(s) can then be called (currently supports matching and IPW) and predicted values are saved to the instance. They can then be used to estimate the ATE, balance groups, and perform EDA (i.e. assess positivity/propensity overlap), etc.

    Parameters
    ----------
    data : pd.DataFrame
        The input data as a pandas dataframe containing X, T, and Y.
    treatment : str
        Name of the treatment column in the input data. Currently only supports binary treatments.
    outcome: str
        Name of the outcome column in the input data. Supports binary and continous outcomes.
    col_names : list
        List of feature names to be included in X (covariates).
    prop_model_algo : str
        The algorithm to use for the propensity model. Currently only supports 'xgb' (XGBoost).
    seed : int
        Random seed for reproducibility.

    Attributes
    ----------
    
    
    
    """

    def __init__(self, data: pd.DataFrame, treatment: str, outcome: str, col_names: list, prop_model_algo='xgb', seed=2025):
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.col_names = col_names
        self.prop_model_algo = prop_model_algo
        self.seed = seed
        self.split_data = self._sample_split()
        self.prop_models, self.prop_scores = None, None
        self.prop_models, self.prop_scores = self._propensity_modeling()
        self.matched_data = None
        self.ate_estimates = {}
        self.cate_estimates = {}

    def _sample_split(self):
        """
        Split the input data into two samples for cross fitting. Stratifies on T or T + Y if binary outcome.
        """
        df = self.data
        treatment = self.treatment
        outcome = self.outcome
        col_names = self.col_names

        #stratify on T only, if outcome is continous
        if df[outcome].nunique() > 2:
            df['stratify_var'] = df[treatment]
        else:
            # Create a combined stratification variable if outcome is binary
            df['stratify_var'] = df[treatment].astype(str) + '_' + df[outcome].astype(str)

        x, t, y, idx = df, df[treatment], df[outcome], df.index
        x_a, x_b, t_a, t_b, y_a, y_b, idx_a, idx_b = train_test_split(
            x, t, y, idx,
            test_size=0.5,
            stratify=x['stratify_var'],
            random_state=self.seed
        )

        x_a = x_a[col_names]
        x_b = x_b[col_names]

        return [(x_a, t_a, y_a, idx_a),(x_b, t_b, y_b, idx_b)]

    def _propensity_modeling(self, params=None, predict_on_same_split=False):
        """
        Cross-fit propensity models and use out of sample predictions as the propensity scores.

        Parameters
        ----------
        params : dict
            Parameters for the propensity model. Use default values unless there is very good reason to change them (i.e. additionally reduce overfitting)
        predict_on_same_split : bool
            Whether to predict on the same split that was used to fit model. Default is False, only override if cross-fitting performance is excessively poor (i.e. AUC < 0.6 or PR-AUC ~= positive class proportion)
        """

        df = self.data
        treatment = self.treatment

        #Cross fit propensity models
        if params is None:
            params = {
                'n_estimators': 1000,
                'max_depth': 4,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 1.0,
                'scale_pos_weight': (df[treatment].shape[0] - df[treatment].sum()) / df[treatment].sum(),
                'min_child_weight': 3.0,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': self.seed,
                'n_jobs': -1,
            }
        else:
            params = params

        if self.prop_model_algo == 'xgb':
            prop_models = []
            for xi, ti, *_ in self.split_data:
                model = xgb.XGBClassifier(**params)
                model.fit(xi.values, ti.values)
                prop_models.append(model)
        
        #Generate out of sample predictions as propensity score and update the stored split data. Default is to predict on the split that wasn't used for training, but can be overriden to predict on the same split if cross-fitting performance is poor (i.e. overfitting due to small data size)

        if not predict_on_same_split:
            #Predict on the other split
            g_hat_a = prop_models[1].predict_proba(self.split_data[0][0].values)[:,1]
            g_hat_b = prop_models[0].predict_proba(self.split_data[1][0].values)[:,1]
        else:
            #Predict on the same split
            g_hat_a = prop_models[0].predict_proba(self.split_data[0][0].values)[:,1]
            g_hat_b = prop_models[1].predict_proba(self.split_data[1][0].values)[:,1]
            
        prop_scores = [g_hat_a, g_hat_b]

        #Check quality of propensity models using AUC and PR-AUC
        print('Positive class proportion in data: ', df[treatment].mean())
        print('Model A PR-AUC: ', average_precision_score(self.split_data[1][1],prop_scores[1]), '| Model A AUC: ', roc_auc_score(self.split_data[1][1],prop_scores[1]))
        print('Model B PR-AUC: ', average_precision_score(self.split_data[0][1],prop_scores[0]), '| Model B AUC: ', roc_auc_score(self.split_data[0][1],prop_scores[0]))

        #Plot propensity score distributions for treated vs untreated
        plt.close()
        fig, ax = plt.subplots()

        x_a = self.split_data[0][0]
        x_b = self.split_data[1][0]
        x_a['g_hat'] = g_hat_a
        x_b['g_hat'] = g_hat_b
        x_a[treatment] = self.split_data[0][1]
        x_b[treatment] = self.split_data[1][1]
        x = pd.concat([x_a, x_b])

        fig.suptitle('Treated vs. Untreated Propensity Distributions')
        sns.kdeplot(data = x[x[treatment]==1], x = 'g_hat',  ax = ax, label = 'treated').set()
        sns.kdeplot(data = x[x[treatment]==0], x = 'g_hat',  ax = ax, color='orange', label = 'untreated')
        plt.legend()
        plt.show()

        x_a.drop(columns=[treatment, 'g_hat'], inplace=True)
        x_b.drop(columns=[treatment, 'g_hat'], inplace=True)

        #return models and scores to initialized instance variables if first time running, otherwise update instance variables with new models and scores for re-runs
        if self.prop_models is None:
            if self.prop_scores is None:
                return prop_models, prop_scores
        else:
            self.prop_models = prop_models
            self.prop_scores = prop_scores

    def match(self, caliper_scale='propensity', caliper=0.03, replace=False, save_matched_data=True):
        """
        Perform matching on the treated and untreated samples using the propensity scores. The caliper parameter is the maximum difference of propensity (either absolute or logit) to be matched, while caliper_scale specifies whether the caliper is an absolute difference in propensity scores or a fraction of the standard deviation of the log-transformed propensity scores. 

        Parameters
        ----------
        caliper_scale : str, optional
            "propensity" (default) if caliper is a maximum difference in propensity scores,
            "logit" if caliper is a maximum SD of log-transformed propensity scores, or
            None for no caliper.
        caliper : float, optional
            Specifies maximum distance (difference in propensity scores or SD of log-transformed propensity scores).
            Default is 0.03.
        replace : bool, optional
            Should individuals from the larger group be allowed to match multiple individuals in the smaller group?
            Default is False.
        save_matched_data : bool, optional
            Whether to save the matched data to the instance variable. Default is True.
        """

        x_a = self.split_data[0][0]
        x_b = self.split_data[1][0]
        x_a['g_hat'] = self.prop_scores[0]
        x_b['g_hat'] = self.prop_scores[1]
        x_a[self.treatment] = self.split_data[0][1]
        x_b[self.treatment] = self.split_data[1][1]
        x_a[self.outcome] = self.split_data[0][2]
        x_b[self.outcome] = self.split_data[1][2]
        df = pd.concat([x_a, x_b]).sort_index()
        
        #set seed here for reproducibility of results
        np.random.seed(self.seed)

        #Create matched data
        treatment = df[self.treatment]
        pscore = df['g_hat']
        match = propensity_functions.Match(treatment,pscore)
        match.create(method='one-to-one', caliper_scale=caliper_scale, caliper=caliper, replace=replace)
        data_matched = propensity_functions.whichMatched(match, df, show_duplicates = False)
        
        x_a.drop(columns=[self.treatment, 'g_hat', self.outcome], inplace=True)
        x_b.drop(columns=[self.treatment, 'g_hat', self.outcome], inplace=True)

        if save_matched_data:
            #Save matched data to instance variable
            self.matched_data = data_matched
        else:
            return data_matched
        
    def check_match_quality(self, features_to_check=None, log_scale=False):
        """
        Check the quality of the matching by comparing the distribution of a feature before and after matching using KDE plots and statistical tests.

        Parameters
        ----------
        data : pd.DataFrame
            Full input data before matching.
        data_matched : pd.DataFrame
            Matched data.
        treatment : str
            Name of the treatment column in the data.
        feature : list
            A list of features to compare. If None (default), all features (covariates) will be compared.
        log_scale : bool, optional
            Whether to log-transform feature list before comparing. Default is False.
        """
        df = self.data
        df_matched = self.matched_data
        treatment = self.treatment

        if features_to_check is None:
            features = self.col_names
        else:
            features = features_to_check

        #Visual inspection
        if log_scale:
            #handle division by 0
            df[features] = df[features].replace(0,0.1)
            df_matched[features] = df_matched[features].replace(0,0.1)

        for feature in features:
            propensity_functions.comparison_plot(df,df_matched, treatment, feature, log_scale=True)
            try: 
                propensity_functions.comparison_stat_tests(df_matched, treatment, feature)
            except:
                pass
    
    def s_learner(self, estimation_type=None, se_type=None,params=None):
        """
        Function for S-learner with IPW.

        """
        df = self.data
        split_data = self.split_data
        prop_scores = self.prop_scores
        features = self.col_names
        treatment = self.treatment
        outcome = self.outcome
        estimation_type = estimation_type
        se_type = se_type

        ate = []
        cate = []

        #Estimation using cross-fitting
        for split in range(len(split_data)):
            print(f'Estimating using split {"A" if split == 0 else "B"} of data')
            not_split = int(not split)
            
            #Get indicies for each split of data to pull them from original df
            idx = split_data[split][-1]
            not_idx = split_data[not_split][-1]
            dfi = df.loc[idx]
            dfi_n = df.loc[not_idx]
            g_hat_i = prop_scores[split]    
            
            # Create S Learner

            #sample weights created using the ω function to calculate inverse propensity weights to further reduce confounding
            print("about create S Learner", flush=True)

            #initialize default params (xgboost)
            if params is None:
                if estimation_type == 'linear':
                    params={}
                else:
                    if df[outcome].nunique() > 2:
                        eval_metric='rmse'
                    else:
                        eval_metric='aucpr'
                    
                    params={
                        'n_estimators': 500,
                        'max_depth': 5,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'scale_pos_weight': (df[treatment].shape[0] - df[treatment].sum()) / df[treatment].sum(),
                        'min_child_weight': 15,
                        'reg_alpha': 0.5,
                        'reg_lambda': 2.0,
                        'eval_metric': eval_metric,
                        'random_state': self.seed,
                        'n_jobs': -1,
                    }
            else:
                params = params
                

            if df[outcome].nunique() > 2:
                if estimation_type == 'linear':
                    s_model = LinearRegression(**params).fit(
                        dfi[features+[treatment]].values, dfi[outcome].values, 
                        sample_weight=causal_functions.ω(g_hat_i, dfi[treatment].values)
                    )
                else:
                    s_model = xgb.XGBRegressor(**params).fit(
                        dfi[features+[treatment]].values, dfi[outcome].values, 
                        sample_weight=causal_functions.ω(g_hat_i, dfi[treatment].values)
                    )
            else:
                if estimation_type == 'linear':
                    s_model = LogisticRegression(**params).fit(
                        dfi[features+[treatment]].values, dfi[outcome].values, 
                        sample_weight=causal_functions.ω(g_hat_i, dfi[treatment].values)
                    )
                else:
                    s_model = xgb.XGBClassifier(**params).fit(
                        dfi[features+[treatment]].values, dfi[outcome].values, 
                        sample_weight=causal_functions.ω(g_hat_i, dfi[treatment].values)
                    )
            # Estimate potential outcomes
            
            if df[outcome].nunique() > 2:
                ##### Regression #####

                # Estimate CATE from total data, using model trained on one half
                x0, x1 = causal_functions.get_potential_outcome_df(df, treatment, features)
                ψi = s_model.predict(x1) - s_model.predict(x0)

                # Estimate CATE from in sample data (i.e. data used to train outcome model above)
                x0, x1 = causal_functions.get_potential_outcome_df(dfi, treatment, features)
                ψ_is = s_model.predict(x1) - s_model.predict(x0)

                # Estimate CATE from out of sample data (i.e. the other fold)
                x0, x1 = causal_functions.get_potential_outcome_df(dfi_n, treatment, features)
                ψ_oos = s_model.predict(x1) - s_model.predict(x0)

            else:
                ##### Classification #####

                # Estimate CATE from total data, using model trained on one half
                x0, x1 = causal_functions.get_potential_outcome_df(df, treatment, features)
                ψi = s_model.predict_proba(x1)[:,1] - s_model.predict_proba(x0)[:,1]

                # Estimate CATE from in sample data (i.e. data used to train outcome model above)
                x0, x1 = causal_functions.get_potential_outcome_df(dfi, treatment, features)
                ψ_is = s_model.predict_proba(x1)[:,1] - s_model.predict_proba(x0)[:,1]

                # Estimate CATE from out of sample data (i.e. the other fold)
                x0, x1 = causal_functions.get_potential_outcome_df(dfi_n, treatment, features)
                ψ_oos = s_model.predict_proba(x1)[:,1] - s_model.predict_proba(x0)[:,1]

                # #Evaluate outcome model performance on out of sample data (i.e. the other fold)
                # oos_predict_proba = s_model.predict_proba(dfi_n[features + [treatment]])[:,1]
                # oos_pr_auc = average_precision_score(dfi_n[outcome].values, oos_predict_proba)
                # print('oos positive class prevalence: ', dfi_n[outcome].mean())
                # print('outcome model PRAUC on other fold: ', oos_pr_auc)
                
            #print results
            print(f'ψ_total = {ψi.mean():.5f} | ψ_in_sample = {ψ_is.mean():.5f} | ψ_out_of_sample = {ψ_oos.mean():.5f} ')

            #adding standard errors - calculated using oos in alignment with cross-fitting
            #approximate calculation
            if se_type == 'approximate':
                se = causal_functions.τ_se(s_model, dfi_n[outcome], dfi_n[dfi_n[treatment]==1][features + [treatment]], dfi_n[dfi_n[treatment]==0][features + [treatment]], ψ_oos, dfi_n[treatment])
                print('oos 95% CIs: ', ψ_oos.mean() - se*1.96, '|', ψ_oos.mean() + se*1.96)
                print('='*30)

            ate.append(ψ_oos.mean())
            cate.append(ψi)
        
        averaged_ate = np.mean(ate)
        averaged_cate = np.mean(cate, axis=0)

        return averaged_ate, averaged_cate


    def ipw_estimation(self, method='s-learner', estimation_type='non-linear', estimator_params=None, se_type='approximate', save_results=True):
        """
        Estimate the (Conditional) Average Treatment Effect using Inverse Propensity Weighting (IPW) and an estimation method (currently supports linear and non-linear meta-learners).

        Parameters
        ----------
        method : str, optional
            The meta-learning method to use for estimation. Default is 's-learner'. Also supports 'x-learner'.
        estimation_type : str, optional
            The type of estimation to fit the data, supports linear and non-linear (default).
        estimator_params : dict, optional
            Params for the estimator. Keep as None unless you have good reason to tune the default params.
        se_type : str, optional
            How standard errors should be calculated. Default is 'approximate', which uses an analytical formula, or 'bootstrap' for bootstrapping estimation.
        save_results : bool, optional
            Whether to save the results to the instance variable. Default is True.
        """
        estimation_type = estimation_type

        #Estimate CATE/ATE with IPW methods:
        if method == 's-learner':
            ate, cate = self.s_learner(se_type=se_type, estimation_type=estimation_type, params=estimator_params)
            if save_results:
                self.ate_estimates['s_learner'] = ate
                self.cate_estimates['s_learner'] = cate
            else:
                return ate, cate
