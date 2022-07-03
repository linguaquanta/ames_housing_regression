# "Data-driven A.I." applied to the Ames Housing regression problem

Throughout the regression analysis presented in this repository, we will be following Andrew Ng's "data-driven AI" paradigm.
That is, we will be sticking with  **ordinary least squares** (OLS) simple linear regression as our model throughout the bulk of the analysis and will be using iterations of its application to aid in the feature selection and construction process. A schematic of this idea is depicted in the
flow diagram below: the output of the simple linear regression is used to modify the original data set to improve downstream applications of the same simple linear regression.

<img src="lin_reg_feat_eng_loop.001.png">


In particular, features are selected using both regression coefficients from fit data and correlations with the target feature, house sale prices. The simplest scheme for feature construction involves generating multiplicative products of the original features. The final set of features is thus a target correlation filtered subset of the originals and the set of all products. By testing products of up to three features, it is found that third-order feature inclusion does not significantly improve the regression performance to justify the combinatorial headache they invite. 
