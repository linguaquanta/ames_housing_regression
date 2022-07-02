import matplotlib.pyplot as plt, numpy as np, pandas as pd, random

def subset_list_by_idxs(arr, subset_idxs):
    ''' Returns subset of list specified by subset index list. '''
    return np.array([arr[i] for i in range(len(arr)) if i in subset_idxs])

def test_train_split(df, test_frac):
    
    num_rows = len(df)
    row_idxs = list(range(num_rows))
    
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1:].values
    
    num_test_rows =  int(np.rint(test_frac*num_rows))

    test_row_idxs = np.random.choice(row_idxs, num_test_rows)
    train_row_idxs = [i for i in row_idxs if i not in test_row_idxs]
    
    X_train = subset_list_by_idxs(X, train_row_idxs)
    y_train = subset_list_by_idxs(y, train_row_idxs)
    
    X_test = subset_list_by_idxs(X, test_row_idxs)
    y_test = subset_list_by_idxs(y, test_row_idxs)
    
    return X_train, y_train, X_test, y_test 

def sort_dict_by_abs_val(in_dict):
    
    ''' Sorts a given dictionary by absolute value of the values. '''
    return dict(sorted(in_dict.items(), key=lambda item: np.abs(item[1])))

def print_null_fracs(df):
    
    ''' Prints all column/feature null fractions for a given dataframe.'''
    
    print('Feature' + '             ' + 'Null Frac')
    for col in list(df.columns):
        null_frac_df = df[col].isnull().value_counts(normalize=True)
        if len(null_frac_df) > 1:
            null_frac = str(np.around(null_frac_df[True], 4))
            print(col + ' '*(20-len(col)) + null_frac)

def normalize_feats(df):
    
    """ Normalizes all features by maxima and returns dict with
        feature names as keys and maxima as values. """
    
    cols = list(df.columns)
    max_vals, max_vals_dict = list(), dict()
    for col in cols:

        max_val = np.max(list(df[col].values))
        max_vals.append(max_val)
        df[col] = df[col].apply(lambda x: x/max_val)

    max_vals_dict = dict(zip(cols, max_vals))
    return max_vals_dict


class LinearRegression:
    
    def __init__(self, fit_intercept=True):
        
        self._fit_intercept = fit_intercept
        self.coeffs_, self.intercept_ = None, None
        
    def fit(self, X, y):
        
        """
        Fit model coefficients. Arguments:
        X: 1D or 2D numpy array 
        y: 1D numpy array
        """
        
        # Check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        # Add bias if fit_intercept is True
        if self._fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        
        # Closed form solution
        xT_x = np.dot(X.T, X)
        inv_xT_x = np.linalg.inv(xT_x)
        xT_y = np.dot(X.T, y)
        coeffs = np.dot(inv_xT_x, xT_y)
        
        # Set attributes
        if self._fit_intercept:
            self.intercept_, self.coeffs_ = coeffs[0], coeffs[1:]
        else:
            self.intercept_, self.coeffs_ = 0.0, coeffs
            
    def predict(self, X):
        
        """
        Output model prediction. Arguments:
        X: 1D or 2D numpy array 
        """
        
        # Check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1,1) 
        return self.intercept_ + np.dot(X, self.coeffs_) 
        
        
class Metrics:
    
    def __init__(self, X, y, model):
        
        self.data, self.target, self.model = X, y, model
        self.dofs_pop_dep_ = X.shape[0] - 1 # Deg of freedom pop dep variable variance
        self.dofs_pop_err_ = X.shape[0] - X.shape[1] - 1 # Deg of freedom pop error variance
        
    def rsdls(self):
        
        ''' Returns the difference between the lin. reg. prediction
            and the true value such that a positive result means the
            prediction has exceeded the ground truth value.'''
        
        return self.model.predict(self.data)-self.target
    
    def frac_rsdls(self):
        
        ''' Returns the fractional difference between the lin. reg. 
            prediction and the true value such that a positive result 
            means the prediction has exceeded the ground truth value.'''
        
        return (self.model.predict(self.data)-self.target)/self.target
    
    def sum_sq_errs(self):
        
        '''Returns sum of squared errors (model vs actual)'''
        sq_errs = (self.target - self.model.predict(self.data)) ** 2
        self.sq_err_ = np.sum(sq_errs)
        return self.sq_err_
        
    def sum_sq_devs_from_avg(self):
        
        '''Returns total sum of squared errors (actual vs avg(actual))'''
        avg_y = np.mean(self.target)
        sq_errs = (self.target - avg_y) ** 2
        self.sum_sq_devs_from_avg_ = np.sum(sq_errs)
        return self.sum_sq_devs_from_avg_
    
    def r_sq(self):
        
        '''Returns calculated value of R^2'''
        self.r_sq_ = 1.0- self.sum_sq_errs()/self.sum_sq_devs_from_avg()
        return self.r_sq_
    
    def adj_r_sq(self):
        
        '''Returns calculated value of adjusted R^2'''
        adj_sum_sq_errs = self.sum_sq_errs()/self.dofs_pop_err_
        adj_sum_sq_devs = self.sum_sq_devs_from_avg()/self.dofs_pop_dep_
        self.adj_r_sq_ = 1.0-adj_sum_sq_errs/adj_sum_sq_devs
        return self.adj_r_sq_
    
    def mean_sq_err(self):
        
        '''Returns calculated value of mean squared error.'''
        self.mean_sq_err_ = np.mean((self.model.predict(self.data)-self.target)**2)
        return self.mean_sq_err_
    
    def pretty_print_stats(self):
        
        '''Returns report of statistics for a given model object'''
        items = (('sse:', self.sum_sq_errs()), 
                 ('sst:', self.sum_sq_devs_from_avg()), 
                 ('mse:', self.mean_sq_err()), 
                 ('r^2:', self.r_sq()), 
                 ('adj_r^2:', self.adj_r_sq()))
        for item in items:
            print('{0:8} {1:.4f}'.format(item[0], item[1]))