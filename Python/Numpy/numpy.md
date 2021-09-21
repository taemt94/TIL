# 2021/09/21
### Counting unique values of numpy array
``` python
import numpy as np

a = np.array([[1, 2, 3], [2, 3, 4]])
unique, counts = np.unique(a, return_counts=True) ## return_counts=True인 경우 unique value의 빈도 수를 리턴
unique_dict = dict(zip(unique, counts))
print(unique_dict)

>>> {1: 1, 2: 2, 3: 2, 4: 1}
```