# 2021/07/09
### OpenCV를 사용한 비디오 재생
``` python
import cv2

cap = cv2.VideoCapture("{Video_directory}")

## openCV version check ##
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

while True:
    try:
        success, img = cap.read()
        
        cv2.imshow("Video", img)

        ## cv2.waitKey(X)의 X는 이미지가 몇 milliseconds 간격으로 디스플레이될 것인지를 의미한다.
        ## 위에서 재생할 비디오는 30fps(비디오마다 다르다.)를 가지므로, 30fps = 30frame/second=30frame/1000ms이다.
        ## 이에 역수를 취하여 1000ms/30frame의 간격으로 이미지를 디스플레이하면, 기존의 비디오와 동일한 속도로 재생할 수 있다.
        key = cv2.waitKey(int(1000/fps))
        if key & 0xFF == ord("q"):
            cv2.destroyAllwindows()
            break
    except:
        pass

cap.release()
```

### 영상에 텍스트(캡션) 추가하기
``` python
import cv2

cap = cv2.VideoCapture("{Video_directory}")

while True:
    try:
        success, img = cap.read()

        ## org: 텍스트 위치, fontFace: 글씨체(옵션 다양함), fontScale: 텍스트 크기,
        ## color: 텍스트 색, thickness: 텍스트 두께
        cv2.putText(img, str("Put text here..."), org=(20, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                                                  color=(255,255,255), thickness=2)

        cv2.imshow("Video", img)
        key = cv2.waitKey(int(1000/fps))
        if key & 0xFF == ord("q"):
            cv2.destroyAllwindows()
            break
    except:
        pass

```