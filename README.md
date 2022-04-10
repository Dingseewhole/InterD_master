# Interpolative Distillation for Unifying Biased and Debiased Recommendation
This is the official pytorch implementation of InterD, a debiasing & biasing method for recommendation system. InterD is proposed in the paper:

[Interpolative Distillation for Unifying Biased and Debiased Recommendation]

by  Sihao Ding, Fuli Feng, Xiangnan He, Jinqiu Jin, Wenjie Wang, Yong Liao and Yongdong Zhang

Published at SIGIR 2022.

## Introduction

InterD is a method that unifies biased and debiased methods to ahcieve strong performance on both normal biased test and debiased test and alleviates over-debiased issue and bias amplification issue in recommendation.

## Environment Requirement

The code runs well under python 3.8.10. The required packages are as follows:

- pytorch == 1.7.1
- numpy == 1.19.1
- scipy == 1.5.2
- pandas == 1.1.3
- cppimport == 20.8.4.2
- tqdm == 4.62.3 
