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

# 2021/07/24
### 영상 저장하기
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

### Default resolutions of the frame are obtained. ###
frame_width = int(caps.get(3))
frame_height = int(caps.get(4))

### fourcc는 영상의 코덱을 설정하는 것으로 'MJPG' 형식으로 하는 것이 가장 일반적이고 안정적이라고 한다.
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video_out = cv2.VideoWriter('./{video_name}.avi', fourcc, fps, (frame_width, frame_height))

while True:
    try:
        success, img = cap.read()

        ### 경험상 imshow()를 하기 전에 write()를 할 때 에러가 발생하지 않았던 것 같다.
        video_out.write(img)
        
        cv2.imshow("Video", img)

        key = cv2.waitKey(int(1000/fps))
        if key & 0xFF == ord("q"):
            cv2.destroyAllwindows()
            break
    except:
        pass

cap.release()
### 영상 저장 객체도 release() 하는 것을 잊지 말아야 한다.
### 이걸 깜빡했다가 하루종일 헤맷던 적이 있다.
video_out.release()
```