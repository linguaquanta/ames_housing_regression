from helper_funcs import find_lack_vars_and_gen_var_dict, print_table
import numpy as np
import pandas as pd

df = pd.read_csv('AmesHousing.csv')
cols = list(df.columns)

var_descr_file = 'VariableDescriptions.txt'
lack_vars, var_descrs = find_lack_vars_and_gen_var_dict(var_descr_file)
fix_var_names = dict(zip(list(var_descrs.keys()), cols))

# Replace `NA` entries w/ `'None'` strings.
for var in lack_vars:
    var = fix_var_names[var]
    df[var] = df[var].apply(lambda x: 'None' if pd.isna(x) else x)

# Although we have replaced the NA entries with 'None' strings and have 
# thus made all corresponding columns completely non-null, there still may 
# be too large a fraction of such entries to justify keeping such columns.

none_thresh = 0.3
none_fracs = [df[fix_var_names[var]].value_counts(normalize=True)['None'] for var in lack_vars]
none_fracs_dict = dict(zip(lack_vars, none_fracs))
lack_thresh_vars = [var for var in lack_vars if none_fracs_dict[var] > none_thresh]
none_thresh_fracs = [none_frac for none_frac in none_fracs if none_frac > none_thresh]
# print_table(['Feature', 'None Frac'], [lack_thresh_vars, none_thresh_fracs])

# We do not immediately drop `FireplaceQu` because it is merely missing
# half of its entries.
drop_cols = ['Alley', 'Pool QC', 'Fence', 'Misc Feature']
df.drop(columns=drop_cols, inplace=True)

lack_thresh_vars = list(set(lack_thresh_vars)-set(drop_cols))
none_fracs = [df[fix_var_names[var]].value_counts(normalize=True)['None'] for var in lack_thresh_vars]
none_thresh_fracs = [none_frac for none_frac in none_fracs if none_frac > none_thresh]
print_table(['Feature', 'None Frac'], [lack_thresh_vars, none_thresh_fracs])

cols = list(df.columns)
bona_fide_nulls = list()
for col in cols:
    
    if len(df[col].isnull().value_counts().index.tolist())==2:
        bona_fide_nulls.append(col)

round_digits = 4

null_fracs_df = df[col].isnull().value_counts(normalize=True)
null_nums_df = df[col].isnull().value_counts(normalize=False)

null_fracs = [str(np.around(null_fracs_df[True], round_digits)) for col in features]
num_nulls = [str(np.around(null_nums_df [True], round_digits)) for col in features]

drop_rows = [df[df[col].isnull()][col].index.tolist() for col in features if df[col].isnull().value_counts(normalize=False)[True] < 25]

col_titles = ['Feature', 'Null Frac', 'Num Null']
cols = [bona_fide_nulls, null_fracs, num_nulls]
print_table(col_titles, cols)

drop_rows = [item for sublist in drop_rows for item in sublist]
df.drop(drop_rows, inplace=True)

null_idxs_dict = {col: df[col].isnull().value_counts().index.tolist() for col in bona_fide_nulls}
null_exist_dict = {col: len(null_idxs_dict[col]) > 1 for col in bona_fide_nulls}
bona_fide_nulls = [col for col in bona_fide_nulls if col in list(df.columns) and null_exist_dict[col]]
print(bona_fide_nulls)