import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def KnearestNeighbors(new_instance,matrix,label,dic):
    classifier=KNeighborsClassifier()
    classifier.fit(matrix,label)
    predicted=classifier.kneighbors(new_instance,n_neighbors=3,return_distance=False)
    for_demo=classifier.kneighbors(new_instance,n_neighbors=3)
    print(f"the distance of the corresponding nearest classes \
\n{for_demo[1]}==>",for_demo[0])
    output=[]
    for i in range(predicted.shape[1]):
        output.append(dic[predicted[0,i]+1])
    return output

    
matrix=np.array([[1,1,0,0,0],
                 [1,1,0,1,1],
                 [1,0,1,0,0],
                 [1,1,1,1,1]]
                )
label=[1,2,3,4]
dic={1:"comedy",2:"action",3:"muzmur",4:"Dynamics"}
new_instance=[[1,1,0,1,1]]
output=KnearestNeighbors(new_instance,matrix=matrix,label=label,dic=dic)
counter=0
for recommend in output:
    print(f"the recommended movie_{counter+1} is ",recommend)
    counter+=1


