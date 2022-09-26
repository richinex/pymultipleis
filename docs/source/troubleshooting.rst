=========================================
Troubleshooting
=========================================

If the standard deviation for certain parameters is higher than 30%

1. Wrong model: Check that the model is correct.

2. Outliers: Remove points in the low-frequency or high-frequency regions which appear due to inductive or non-linearity distortions

3. Wrong weighting: If the deviation in the low frequency region is more significant than in the high frequency region, change weighting from modulus to unit. 
   Unit weighting favours low frequencies against high frequencies or try other weighting options.

4. Redundant parameter: Fix the parameter with the largest deviation manually by setting its smoothing factor to Inf

5. Bad starting values: try changing initial guesses manually.

6. Low number of iterations: Increase the number of iterations by changing the value of the n_iter argument of fit_deterministic

7. Wrong choice of optimizer: Try fit_stochastic instead of fit_deterministic

