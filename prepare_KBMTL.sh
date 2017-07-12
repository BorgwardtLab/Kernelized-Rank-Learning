#!/bin/bash

git clone https://github.com/mehmetgonen/bmtmkl.git
git clone https://github.com/ewiger/yamlmatlab.git

cp bmtmkl/bayesian_multitask_multiple_kernel_learning_train.m KBMTL/bayesian_multitask_multiple_kernel_learning_train.m
cp bmtmkl/bayesian_multitask_multiple_kernel_learning_test.m KBMTL/bayesian_multitask_multiple_kernel_learning_test.m
