# Evolving Separating References for Time Series Classification (ESR)

This repository contains the code accompanying the paper, "[Evolving Separating References for Time Series Classification](https://epubs.siam.org/doi/pdf/10.1137/1.9781611975321.28)" (Xiaosheng Li and Jessica Lin, SDM 2018). This paper presents that sequences that are very different from the patterns in the time series can be used as references to classify the time series effectively. The proposed approach is especially suitable for the situations where not much labeled data is available.

## Datasets

To run the experiments in the paper, one needs to download the UCR benchmark datasets from: www.cs.ucr.edu/~eamonn/time_series_data/. Then unzip the downloaded file and put the data folder into the same directory as the source code.

## Mex Function

The code uses mex function to compute distance to fasten the program. The C mex source code (individualDistance_c.c and normalizedIndividualDistance_c.c) is included. The compiled mex files for 64-bit Linux and Windows are provided. For other operation systems, the user needs to compile the two C files into mex functions by using the "mex" command in MATLAB.

## To Run the Code

The user needs to specify the "dataset" to run on in Line 18 of `ESR.m` and "parameter" in Line 19 of `ESR.m`. The parameter can take the value of 1 or 2, corresponding to S1 or S2 in the paper.

After specifying the dataset and parameter, run `ESR.m` in MATLAB will run the code. The code will train ESR on the training data and perform classification on the testing data, output the testing error rate.

## Parallelization

It is recommended to parallelize the evaluation of the individuals in the population, which can significantly fasten the program (especially on medium and large size data). This can be done by uncommenting Line 108-109, and last line of `ESR.m`, also uncommenting Line 16-19 of `evaluation.m` and commenting Line 12-13 of `evaluation.m`. The number in Line 108 of `ESR.m` corresponds to the number of CPU cores to use in the machine or cluster.

## Citation
```
@inproceedings{li2018evolving,
  title={Evolving separating references for time series classification},
  author={Li, Xiaosheng and Lin, Jessica},
  booktitle={Proceedings of the 2018 SIAM international conference on data mining},
  pages={243--251},
  year={2018},
  organization={SIAM}
}
```

