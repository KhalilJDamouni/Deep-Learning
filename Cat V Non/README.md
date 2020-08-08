Cat V NonCat README

Testing DeepLearning with C. Trying without GMP, with GMP, and with linear algebra operations to quantize vectorization performance improvements.

DeepLearning V1: Translating "Logistic_Regression_with_a_Neural_Network_mindset_v6a" by deeplearning.ai from python to C
Comments:
    Values go to 0. Thought it was because more precision was required, but I had just made a mistake.

DeepLearning V2: Fixing V1, using less hardcoded values. Works for any size train and test set for 64*64*3 pictures.
Comments:
    Near exact performance as python.
    After clean up will test for runtime.
