# Local Periodicity-Based Beat Tracking for Expressive Classical Piano Music

This is the repo for the paper titled
"*Local Periodicity-Based Beat Tracking for Expressive Classical Piano Music*".

[ **| Paper |** ](https://)[ **Demo** ](https://sunnycyc.github.io/plpdp4beat-demo/)[ **| Code |** ](https://github.com/SunnyCYC/plpdp4beat/)

## Abstract
To model the periodicity of beats, state-of-the-art beat tracking systems use “post-processing trackers” (PPTs) that rely on several empirically determined global assumptions for tempo transition, which work well for music with steady tempo. For expressive classical music, however, these assumptions can be too rigid. With two large datasets of Western classical piano music, namely the Aligned Scores and Performances (ASAP) dataset and a dataset of Chopin’s Mazurkas (Maz-5), we report on experiments showing the failure of existing PPTs to cope with local tempo changes, thus calling for new methods. In this paper, we propose a new local periodicity-based PPT, called predomi-nant local pulse-based dynamic programming (PLPDP) tracking, that allows for more flexible tempo transitions. Specifically, the new PPT incorporates a method called “predominant local pulses” (PLP) in combination with a dynamic programming (DP) component to jointly consider the locally detected periodicity and beat activation strength at each time instant. Accordingly, PLPDP accounts for the local periodicity, rather than relying on a global tempo assumption. Compared to existing PPTs, PLPDP particularly enhances the recall values at the cost of a lower precision, resulting in an overall improvement of F1-score for beat tracking in ASAP (from 0.473 to 0.493) and Maz-5 (from 0.595 to 0.838).
## Usage
In this repo we include one recording from the ASAP dataset [1, 2] as an example to demonstrate the usage of the [inference code](https://github.com/SunnyCYC/plpdp4beat/blob/main/inference.py). Users may run the [inference code](https://github.com/SunnyCYC/plpdp4beat/blob/main/inference.py) directly to see how it generates both estimated beats and qualitative results using the post-processing trackers (i.e., SPPK, DP, HMM, HMMT0, and PLPDP) considered in the paper.

## Extended Experiments
### Kernel Sizes of PLP
The following figure shows the beat tracking F1-score of single kernel PLPDP using kernel sizes from 1-20 seconds. The F1-scores of combined kernel PLPDP (i.e., $\kappa=1, 3, 5$ seconds) are also plotted (dashed lines) for comparison. It can be seen that for both ASAP and Maz-5, kernel sizes larger than five seconds do not influence the F1-score much. This could be the results of multiple different factors. For example, longer kernels may perform better in regions with stabler tempo (as they capture more information regarding the local periodicity) but perform worse in regions with larger tempo variations. Moreover, in regions with relatively stable tempo, using kernel size of five or 15 seconds may not make big difference. However, it would be difficult to analyze the relation between suitable kernel size and distributions of stable/unstable regions in datasets. Therefore, we empirically consider kernel sizes smaller than five seconds. On the other hand, it can also be seen that while $\kappa=1$ performs the worst for both datasets, Maz-5 prefers $\kappa=3$ and ASAP prefers $\kappa \geq 5$. This is mainly because the detection of local periodicity requires the kernels to cover an appropriate number of peaks. When the kernel is small (e.g., one second), it is hard to detect and periodicity in regions with slower tempo. Besides, musical pieces in Maz-5 (mean tempo 125 BPM) are generally faster than ASAP (mean tempo 107 BPM), which explains why $\kappa=2, 3$ can be preferred in Maz-5 rather than in ASAP. 

Finally, as illustrated in our paper Section III-B, despite that $\kappa=1$ does not perform well independently, it is quite useful to reduce artifacts. We therefore empirically adopt the combination of $\kappa=1, 3, 5$ seconds without further investigating other combinations.

<img src="https://user-images.githubusercontent.com/60595988/201060398-86795634-6e21-4de8-b711-076d7816f460.png" alt="Cover" width="50%"/>



### Grid Search of HMM Tempo Transition Lambda
Grid search experiments are conducted to investigate the performance of HMM using tempo transition lambda from 0--100 with a step size of five. The following figure shows the result for real datasets. The results of PLPDP are also plotted as horizontal dashed lines for comparison. 

We can see that despite that HMMs with lambda $=5-25$ indeed perform better than HMMT0, the best HMMs still perform worse than PLPDP in both datasets. We can also see that though both datasets are "expressive classical music", the preferred lambdas are very different (i.e., Maz-5: lambda=5, ASAP: lambda = 90), indicating the dramatic difference between expressive musical pieces.

<img src="https://user-images.githubusercontent.com/60595988/201060473-8a15490c-8b95-4ee7-9982-1e0063a7c988.png" alt="Cover" width="50%"/>


On the other hand, the Figure below shows the grid search results of synthetic datasets using different tempo transition lambdas. It can be seen that in synthetic experiments, for both ASAP and Maz-5, lambda $=5$ works the best. As the value of lambda increases, the performance of HMM for both datasets decreases monotonically in different slopes.

<img src="https://user-images.githubusercontent.com/60595988/201060529-aa79d49e-2a51-4d7e-af02-a900de4f026f.png" alt="Cover" width="50%"/>

We summarize the main ideas as follows:
* The results of Figures 3 and 4 reveal the remarkable influence of the tempo transition lambda on the performance of HMM for expressive music. They also validate PLPDP’s superiority over HMMs in both datasets when using real activation functions.
* From the different preference of lambda for ASAP and Maz-5 in real activation experiment, it can be seen that “adjust the lambda for the characteristics of the data” may be impractical in real use cases. As the characteristics of expressive classical music may vary dramatically, the optimal lambda for a dataset may also be problematic for individual pieces.
* Again, note that similar “global” vs. “local” dilemma also appears within individual expressive musical pieces. It is common that an expressive musical piece has regions with stable tempo and regions with dramatic tempo changes. And a “best lambda” for that musical piece may be problematic for specific regions.
* The synthetic experiments also reveal the impracticality of adjusting the lambda based on real activation. As the real ASAP prefers lambda = 90 which is dramatially different from what synthetic ASAP prefers (i.e, lambda= 5), it can be seen that the interaction between imperfect beat activation and limited tempo related assumptions make the error analysis or parameter tuning difficult.
* This way, we can see again the value of synthetic experiments. The “perfect” synthetic activation allow us to exclude factors of imperfect beat activation and see clearer the limitations of the post-processing trackers (PPTs).
* As we have already known the PPTs work based on different assumptions (i.e., “local periodicity” vs. “global tempo transition settings”), and our goal in this work is to investigate the limitations of PPTs, rather than comparing them after parameter optimization, we decided to keep the original baselines (i.e., HMMT0, HMM) and put all these grid search discussions in our repository as a supplement.

### More Evaluation Metrics
As existing conventional evaluation metrics generally assume a fixed relation between estimated beats and reference beats (e.g., double tempo or onbeat throughout a whole sequence), they are not able to reflect the "metric-level switching" behaviors of PPTs for expressive music. We have recently proposed an analysis method to compensate existing metrics[3]. Briefly, our proposed analysis method, annotation coverage ratio (ACR), calculate for each musical piece how the reference beats are "covered" (i.e., detected) by the estimated beats. For example, if the estimated beats switch to double tempo of reference beats for half of the time, while existing metric like AMLt would give a score of 0.5, ACR can reveal 0.5 for onbeat, 0.5 for double tempo, and 1.0 for any tempo. 

From Table 1, we can see that potential inconsistency between existing conventional metrics. For example, for Maz-5, HMMT0 achieves higher F1-score (0.595) than DP does (0.488), while DP achieves much higher CMLt and AMLt than HMMT0. With ACR, the inconsistency became explainable. As HMMT0 switches between subharmonic tempi (i.e., half, third, quarter), the scores of CMLt and AMLt can only be low. However, due to its high precision, HMMT0 still gets a high F1-score. It can also be observed that none of existing PPTs really learn to correctly determine the local tempo of expressive musical pieces, and different PPTs behave differently based on their assumptions. The ACR results explains the high recall of PLPDP and high precision of HMMs. As PLPDP relies on local periodicity calculated based on local windowed activation peaks, for ASAP dataset with large amount of non-beat activation peaks, PLPDP is prone to detect faster harmonic tempi, which results in high recall and low precision. On the other hand, HMMs tend to tap slower in both datasets, therefore achieve higher precision and lower recall.

![image](https://github.com/SunnyCYC/plpdp4beat/assets/60595988/92693234-f4a9-4fa7-b915-88d20191d9e3)


## Reference
*[1] F. Foscarin, A. McLeod, P. Rigaux, F. Jacquemard, and M. Sakai,“ASAP: A dataset of aligned scores and performances for piano transcription,” in Proc. Int. Soc. Music Inf. Retr. Conf., 2020, pp. 53.*

*[2] https://github.com/fosfrancesco/asap-dataset*

*[3] C. Y. Chiu, M. Müller, M. E. P. Davies, A. W. Y. Su and Y. H. Yang, "An Analysis Method for Metric-Level Switching in Beat Tracking," in IEEE Signal Processing Letters, vol. 29, pp. 2153-2157, 2022.*
