# Predominant Local Pulse-based Dynamic Programming Beat Tracker

This is the repo for the paper titled
"*Local Temporal Expectation-based Beat Tracking for Expressive Classical Music*".

[ **| Paper |** ](https://)[ **Demo** ](https://sunnycyc.github.io/plpdp4beat-demo/)[ **| Code |** ](https://github.com/SunnyCYC/plpdp4beat/)

## Abstract
To model the periodicity of beats, state-of-the-art beat tracking systems use “post-processing trackers” (PPT) that rely on several globally and empirically determined tempo transition assumptions, which have been shown to work well for tracking music with steady tempo. For expressive classical music, however, these assumptions might be too rigid. With two large datasets of classical music, Maz-5 and ASAP, we report experiments suggesting the failure of existing PPTs to cope with local tempo changes, thus calling for a new method. Motivated by findings in cognitive neuroscience, we propose a new local temporal expectation-based PPT, called PLPDP, that allows for more flexible tempo transition. Specifically, the new PPT incorporates a method called “predominant local pulses” and a dynamic programming tracker to jointly consider the locally detected periodicity and beat activation strength at each time instant. Accordingly, PLPDP accounts for the local periodicity consistency, rather than assuming global periodicity consistency. Compared to existing PPTs, PLPDP improves the F1 score for beat tracking in Maz-5 and ASAP by 21.9% and 3.1%, respectively.

## Usage
In this repo we include one recording from the ASAP dataset [1, 2] as an example to demonstrate the usage of the [inference code](https://github.com/SunnyCYC/plpdp4beat/blob/main/inference.py). Users may run the [inference code](https://github.com/SunnyCYC/plpdp4beat/blob/main/inference.py) directly to see how it generates both estimated beats and qualitative results using the post-processing trackers (i.e., SPPK, DP, HMM, HMMT0, and PLPDP) considered in the paper.



## Reference
*[1] F. Foscarin, A. McLeod, P. Rigaux, F. Jacquemard, and M. Sakai,“ASAP: A dataset of aligned scores and performances for piano transcription,” in Proc. Int. Soc. Music Inf. Retr. Conf., 2020, pp. 53.*

*[2] https://github.com/fosfrancesco/asap-dataset*
