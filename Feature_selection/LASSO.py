import pandas as pd
import numpy as np
import tools as utils
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso	
from sklearn.preprocessing import scale
path1 = r'/Users/ada/Desktop/xgboost/no.2/extraction0720.csv'
data_2=np.loadtxt(path1, delimiter=',')
label_1=np.ones((int(178),1))#Value can be changed
label_2=np.zeros((int(226),1))
label=np.append(label_1,label_2)
shu=scale(data_2)
lasso = Lasso(alpha=0.04,tol=0.1)									
lasso.fit(shu, label)									
model = SelectFromModel(lasso,prefit=True)
data_1 = model.transform(shu)
X=data_1
y=label
sepscores = []
cv_clf =XGBClassifier(
              base_score=0.5, 
              colsample_bylevel=1, 
              colsample_bytree=0.8,
              gamma=0.4,#
              learning_rate=0.1, #
              max_delta_step=0,
              max_depth=4,#
              min_child_weight=1, 
              missing=None,
              n_estimators=100, #
              nthread=-1,
              objective='binary:logistic', #
              reg_alpha=1, 
              reg_lambda=1,
              scale_pos_weight=1, 
              seed=1234, #
              silent=True, 
              subsample=1
            )
skf= StratifiedKFold(n_splits=5)
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
for train, test in skf.split(X,y): 
    y_train=utils.to_categorical(y[train])
    hist=cv_clf.fit(X[train], y[train])
    y_score=cv_clf.predict_proba(X[test])
    yscore=np.vstack((yscore,y_score))
    y_test=utils.to_categorical(y[test]) 
    ytest=np.vstack((ytest,y_test))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= utils.categorical_probas_to_classes(y_score)
    y_test_tmp=y[test]
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('XGBoost:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))

result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores
row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='red',
lw=lw, label='Lasso (AUC = %0.4f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")

plt.savefig('rocSelect.jpeg',format='jpeg',dpi=2000)
plt.show()




