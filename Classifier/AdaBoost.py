import pandas as pd
import numpy as np
import tools as utils
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from imblearn.over_sampling import BorderlineSMOTE
import matplotlib as mpl
import matplotlib
#from collections import Counter
path1 = r'/Users/ada/Desktop/xgboost/no.2/newlasso0720.csv'
data1=np.loadtxt(path1, delimiter=',')
label_1=np.ones((int(178),1))#Value can be changed
label_2=np.zeros((int(226),1))
label=np.append(label_1,label_2)
smo = BorderlineSMOTE(kind='borderline-1',sampling_strategy={0:246,1:246}) #kind='borderline-2'
X_smo, y_smo = smo.fit_sample(data1, label)
X=X_smo
y=y_smo

sepscores = []
cv_clf =AdaBoostClassifier(
             base_estimator=None, 
             n_estimators=250
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
    print('AdaBoost:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
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
lw=1
plt.plot(fpr, tpr, color='orange',
lw=lw, label='AdaBoost (AUC = %0.4f)' % auc_score)
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
lw=1
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
ax = plt.gca()
plt.tick_params(labelsize=8) 
labels = ax.get_xticklabels() + ax.get_yticklabels()
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.03, 1.03])
plt.ylim([0.0, 1.03])
plt.xlabel('False positive rate',font)
plt.ylabel('True positive rate',font)
font_L = {'family': 'Times New Roman', 'weight': 'normal', 'size':9.5}
legend = plt.legend(prop=font_L,loc="lower right")
plt.savefig('rocClass.jpeg',format='jpeg',dpi=2000)
plt.show()

