# Cat V NonCat README

### Testing DeepLearning with C. Trying without and with linear algebra operations to quantize vectorization performance improvements.


|DeepLearningVx | Use | Description | Runtime|
|----------------|-----|-------------|--------|
|V0 | Baseline Python comparison| Original "Logistic_Regression_with_a_Neural_Network_mindset_v6a" by deeplearning.ai in Python| 25.80s |
|V1 | Ensuring the neural network achieves the same result in C | Translating "Logistic_Regression_with_a_Neural_Network_mindset_v6a" by deeplearning.ai from python to C | N/A |
|V2 | Baseline C comparison (No external libraries) |  Fixing V1, using less hardcoded values. Works for any size train and test set for 64,64,3 pictures. | N/A |
|V3 | Vectorized approach with GLS_BLAS for comparison | Neural Network in C using GSL_BLAS library. Translating V2 into a vectorized approach for comparison. | 11.38s |

## Runtime Measurement Method
- Average of 20 runs.
- Used a MacBook Air (13-inch, Early 2015)

> Comments:
> Near exact performance as python.
> Runtime calculations in V0 and V2. Will be run multiple times then calculate average for comparison.
