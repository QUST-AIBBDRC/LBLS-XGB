##LBLS-XGB: a computional method for identification of cancerlectins using XGBoost with multi-information fusion

##LBLS-XGB uses the following dependencies:
(1)Python 3.7.4
(2)numpy
(3)scipy
(4)scikit-learn
(5)pandas

##Guiding principles: **The dataset contains both training dataset and independent test set.

**Feature extraction
   CTriad.py implements CT.
   DC_data.m and Dipeptide.m implement DC.
   Dipeptide.m implements g-gap DC.
   PSSMmaker and  PSSMtest implement PsePSSM.
   PWAAC.m and PWAAC_DATA.m implement PWAAC.
** Dimensional reduction:
   LASSO.py and tools.py implement LASSO.
** borderline-SMOTE:
   borderline-SMOTE.py implements borderline-SMOTE.
** Classifier:
   Adaboost.py implements Adaboost.
   ET.py implements ET.
   GTB.py implements GTB.
   KNN.py implements KNN.
   NB.py implements NB.
   RF.py implements RF.
   SVM.py implements SVM.
   XGBoost.py implements XGBoost.
** dataset:
   The training_dataset.txt contains the data of the training dataset.  
   The independent_dataset.txt contains the data of the independent test set.  