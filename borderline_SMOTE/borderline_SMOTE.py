import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
path1 = r'/Users/ada/Desktop/xgboost/no.2/newlasso0720.csv'
data1=np.loadtxt(path1, delimiter=',')
label_1=np.ones((int(178),1))#Value can be changed
label_2=np.zeros((int(226),1))
label=np.append(label_1,label_2)
smo = BorderlineSMOTE(kind='borderline-1',sampling_strategy={0:246,1:246}) 
X_smo, y_smo = smo.fit_sample(data1, label)
X=X_smo
y=y_smo
