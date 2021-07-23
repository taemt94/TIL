# 2021/07/23
### CategoricalCrossentropy vs. SparseCategoricalCrossentropy
- CategoricalCrossentropy()는 label이 각각의 샘플마다 원핫벡터로 표현되어 있을 때 사용하는 loss이다.
- 사용법은 아래와 같다.
- 아래의 두 예시는 3개의 클래스를 가지는 데이터의 loss를 계산한다.
  ``` python
  import numpy as np

  y_true = np.array([[1, 0, 0], [0, 0, 1]])
  y_pred = np.array([[0.90, 0.05, 0.05], [0.1, 0.1, 0.8]])
  cce = tf.keras.losses.CategoricalCrossentropy()
  cce(y_true, y_pred).numpy()

  >>> 0.16425202786922455
  ```
- 아래의 예시는 이미지에 대한 예시를 위해 임의의 값을 설정하여 계산한 것이다.
- y_true는 각각 픽셀의 값이 원핫벡터로 이루어져 있는 형태이다.
  ``` python
  y_true = np.zeros([128, 128, 3], dtype=np.float32)
  y_true[:, :, 1] = 1
  y_pred = np.zeros([128, 128, 3])
  y_pred[:, :, 0] = 0.05
  y_pred[:, :, 1] = 0.9
  y_pred[:, :, 2] = 0.05
  cce = tf.keras.losses.CategoricalCrossentropy()
  cce(y_true, y_pred).numpy()

  >>> 0.10536051541566849
  ```

- SparseCategoricalCrossentropy()는 label이 원핫벡터가 아닌 클래스에 해당하는 인덱스로 이루어져 있을 때 사용하는 loss이다.
- 사용법은 아래와 같다.
- 결과를 보면, label의 표현 방식만 다를 뿐, CategoricalCrossentropy()와 계산 과정은 동일하므로 같은 결과가 나옴을 알 수 있다.
  ``` python
  import numpy as np

  y_true = np.array([0, 2])
  y_pred = np.array([[0.90, 0.05, 0.05], [0.1, 0.1, 0.8]])
  scce = tf.keras.losses.SparseCategoricalCrossentropy()
  scce(y_true, y_pred).numpy()

  >>> 0.16425202786922455
  ```

  ``` python
  y_true = np.ones([128, 128, 1], dtype=np.float32)
  y_pred = np.zeros([128, 128, 3])
  y_pred[:, :, 0] = 0.05
  y_pred[:, :, 1] = 0.9
  y_pred[:, :, 2] = 0.05
  scce = tf.keras.losses.SparseCategoricalCrossentropy()

  >>> 0.10536051541566849
  ```