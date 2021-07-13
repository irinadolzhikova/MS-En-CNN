Subject-independent (SI) classification is a major area of investigation in Brain-Computer Interface (BCI) that aims to construct classifiers of users' mental states based on collected electroencephalogram (EEG) of independent subjects.

'test_MS_En_CNN.py' file shows the script for evaluation of the ensembles with DeepConvNet-based base classifiers.

See the references: [1] K. Zhang, N. Robinson, S.-W. Lee, and C. Guan, “Adaptive transferlearning for eeg motor imagery classification with deep convolutionalneural network,” Neural Networks, vol. 136, pp. 1–10, 2021.

[2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730

'.pt' files are examples of the base classifier models that were trained based on [1] using 3fold-CV and are saved as S_{test_subject}_cv{K}.pt, where {test_subject} is the held-out subject that is used for SI evaluation, {K} is = 0,1,2 is the K in 3-fold CV.

Examples of the base classifiers for test subject 46 and 47 are provided to reproduce the results.

Subject-independent (SI) decoding accuracy results for each test_subject could be seen in   Table S1 of "SI_evaluationResults_for_each_test_subject.pdf" file.

Table S1 shows the subject-independent decoding accuracy results for each individual subject for each of the
following scenarios: 1) each phase (offline and online) and session (s1 and s2) separately (columns 1-4); 
2) for pooled data from the online and the offline phases,while keeping sessions separate (columns 5-6); and 
3) for a pooled data across all phases and sessions (column 7). 
These results are presented for the 54 subject based motor imagery dataset (MI-Dataset) by using 
Multi-Subject Ensemble CNN (MS-En-CNN).
