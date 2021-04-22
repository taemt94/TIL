# 2021/03/30
- 실차 프로젝트를 진행하면서 `ROS`를 공부하게 되었다.
- 차량의 CAN 데이터나 카메라 및 라이다 데이터를 ROS를 통해 송/수신하기 위한 인터페이스를 구축하려한다.
- 그러기 위해서는 일단 ROS를 사용할 줄 알아야 하기 때문에 연습 겸 우분투가 설치되어 있는 노트북에 ROS 설치부터 시작해보려고 한다.  
  
  [ROS installation](http://wiki.ros.org/ROS/Installation)
- ROS 설치는 위의 링크를 통해 확인할 수 있다.
- 우분투 버전 별로 지원하는 ROS 버전이 다르므로 이것부터 확인해보아야 한다.
- 현재 내가 사용하는 우분투는 18.04이기 때문에 ROS Melodic을 설치하였다.  
[ROS environment setup](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment)
- 설치가 완료되면 위의 링크를 통해 ROS 환경 설정을 진행한다.
- 설치 과정에서 중요한 내용만 간단하게 아래로 요약한다.
- 아래의 명령어는 설치가능한 모든 ROS 패키지를 보여준다.
  ```
  $ apt search ros-melodic
  ```
- ROS에서는 catkin이라는 ROS 전용 빌드 시스템을 사용한다고 한다.
- 이를 사용하기 위해서는 아래와 같이 catkin 작업 폴더를 생성하고 초기화해야 한다
  ```  
  $ mkdir -p ~/catkin_ws/src
  $ cd ~/catkin_ws/
  $ catkin_make
  ```
- `catkin_make`를 한 후 `ls` 명령어를 치면 src 폴더 외에 build 폴더와 devel 폴더가 생성된다.
- catkin 빌드 시스템의 빌드 관련 파일은 build 폴더에, 빌드 후 실행 관련 파일은 devel 폴더에 저장된다고 한다.

```
$ roscore
```
- `roscore`는 모든 ROS 시스템을 관할하는 명령어이다.
```
$ rosrun turtlesim turtlesim_node
```
- `rosrun`은 노드를 하나 실행시키는 명령어이다.
- 다음으로 오는 `turtlesim`은 패키지명이다.
- 마지막의 `turtlesim_node`는 하나의 노드이다.

<http://wiki.ros.org/msg>  
<http://wiki.ros.org/common_msgs>
- 메세지에 대한 정보가 있는 링크로, ROS를 사용하기 위해서는 위의 링크를 통해 메세지 정보를 자세히 들여다볼 필요가 있다.

<http://wiki.ros.org/std_msgs>
- 단순 자료형의 메세지 정보를 알려주는 링크이다.

```
$ rostopic list
```
- 현재 주고 받고 있는 토픽의 리스트를 보여준다.
```
$ rostopic echo [topic name]
```
- [topic name]에 해당하는 메세지를 실제로 보여준다.
```
$ rostopic info [topic name]
```
- 메세지의 type, publisher, subscriber 등 메세지의 기본 정보를 알려준다.
- 메세지 type을 검색해보면 해당 메세지를 자세히 알아볼 수 있다.
```
$ rqt
```
- 플러그인 방식의 ROS 종합 툴이다.
- `ros_image_view`, `rqt_graph`, `rqt_plot`, `rqt_bag` 등과 같이 다양한 플러그인이 있고, 이를 통해 다양한 데이터를 시각화할 수 있다.

```
$ export ROS_MASTER_URI=http://[MASTER IP]:11311
$ export ROS_HOSTNAME=[NODE IP]
```
- `ROS_MASTER_URI`는 마스터에 해당하는 IP와 포트 번호를 입력하면 된다.
- `ROS_HOSTNAME`은 노드에 해당하는 IP를 입력하면 된다.
- 이렇게 하면 예를 들어 `ROS_MASTER_URI`에 해당하는 기기에서 촬영하고 있는 영상을 `ROS_HOSTNAME`에 해당하는 기기에서 볼 수 있다.
- 주의할 점은 `ROS_MASTER_URI`와 `ROS_HOSTNAME`가 같은 네트워크에 접속하고 있어야 한다는 점(IP 주소의 앞 3자리가 같으면 된다.)이다.

# 2021/04/04
### ROS 패키지 생성 및 topic 송수신
- ROS에서는 단방향 통신일 때 `topic`이라는 메세지 통신을 사용한다.
- 이 때 송신하는 노드를 `publisher`, 수신하는 노드를 `subscriber`라고 부른다.
#### 1. 패키지 생성
```
$ cd ~/catkin_ws/src
$ catkin_create_pkg ros_tutorials_topic message_generation std_msgs roscpp
$ cd ros_tutorials_topic
$ ls
include       -> 헤더파일 폴더
src           -> 소스코드 폴더
CMakeLists.txt -> 빌드 설정 파일
package.xml   -> 패키지 설정 파일
```
- `catkin_create_pkg` 명령어는 ROS 패키지를 생성해주는 명령어이다.
- 단순히 패키지 폴더 내에 `include` 폴더, `src` 폴더, `CMakeLists.txt` 및 `package.xml`을 생성해주는 명령어이기 때문에 직접 손으로 해당 폴더 및 파일들을 생성하는 것과 똑같지만, 하나하나 직접 생성하는 것보다는 훨씬 편하다.

#### 2. 패키지 설정 파일(package.xml) 수정
```
$ gedit package.xml
```
- `package.xml` 파일을 열어 패키지 설정 파일을 수정한다.
- `package.xml`은 ROS의 필수 설정 파일 중 하나로, 패키지 정보를 담은 xml 파일이다.
- ROS 공식 패키지가 되면, wiki 페이지에 이 파일의 내용들이 들어가게 되므로 잘 작성하는 것이 매우 중요하다.
- 참고로 xml은 extended mark of language의 약자로 태그 기반의 문서라고 하지만, 아직 잘 알지 못하므로 필요 시 추후에 더 공부하기로 한다.
- 패키지를 생성하면 `package.xml` 파일에 기본적인 세팅이 되어 있으므로 필요한 것을 사용해도 무방하다.
``` xml
<?xml version="1.0"?>
<package format="2">

## 패키지 이름으로, 패키지 생성 시 자동으로 입력된다.
<name>ros_tutorials_topic</name> 

## 버전에 대한 tag로, 공식 패키지가 되었을 때는 매우 중요한 요소 중 하나이다(업데이트할 때마다 올라가는 버전이 이것이다.).
## 숫자는 0.1.0부터 시작하던 0.0.1부터 시작하던 사용자 임의로 지정하면 된다.
<version>0.1.0</version> 

## 패키지에 대한 간략한 설명을 작성한다.
## 보통 2-3줄 정도가 적당하다. 
<description>ROS turtorial package to learn the topic</description> 

## 개발 시 적용되는 라이센스를 작성한다.
<license>Apache 2.0</license>

## author와 maintainer 태그는 개발자 및 유지 보수자를 작성하는 태그로, 인원이 다수일 경우 아랫줄에 추가해주면 된다.
<author email="pyo@robotis.com">Yoonseok Pyo</author>
<maintainer email="pyo@robotis.com">Yoonseok Pyo</maintainer>

## url 태그는 패키지에 대한 설명, 소스 코드 레포, 버그를 전달받는 주소 등을 작성하는 태그이다.
<url type="website">http://www.robotis.com</url>
<url type="repository">https://github.com/ROBOTIS-GIT/ros_tutorials.git</url>
<url type="bugtracker">https://github.com/ROBOTIS-GIT/ros_tutorials/issues</url >

## buildtool_depend는 패키지를 빌드할 때 사용할 명령어를 작성하는 태그이다.
## 예전에는 rosbuild라는 빌드 명령어도 있었지만, 현재는 거의 catkin 명령어만 사용한다고 한다.
## ROS2에서는 또 다른 빌드 명령어가 있다고 하니 ROS2를 쓸 때 알아보기로 한다.
<buildtool_depend>catkin</buildtool_depend>

## depend는 의존성 패키지를 작성하는 태그로, 패키지 상에서 사용할 기존에 있는 패키지들을 작성한다.
<depend>roscpp</depend>
<depend>std_msgs</depend>
<depend>message_generation</depend>
<export></export>
</package>
```
- 위와 같이 패키지 설정 파일을 작성한 후 창을 닫는다.

#### 3. 빌드 설정 파일(CMakeLists.txt) 수정
```
$ gedit CMakeLists.txt
```
- `CMakeLists.txt` 파일을 열어 빌드 설정 파일을 수정한다.
- `CMakeLists.txt`은 작성한 노드(소스코드)를 빌드하며 실행 파일을 생성할 때 어떤 식으로 하라는 내용을 담고 있는 파일이다.
``` cmake
## cmake 최소 빌드 버전을 작성한다.
cmake_minimum_required(VERSION 2.8.3)

## 패키지 명을 작성하는 부분으로 철자가 틀리면 빌드가 안되므로 주의한다.
project(ros_tutorials_topic)

## 캐킨 빌드를 할 때 요구되는 구성요소 패키지이다.
## 의존성 패키지는 message_generation, std_msgs, roscpp이며 이 패키지들이 존재하지 않으면 빌드 도중에 에러가 난다.
find_package(catkin REQUIRED COMPONENTS message_generation std_msgs roscpp)

## 메세지 선언: MsgTutorial.msg
## Topic에 해당하는 메세지 파일로 뒤에 해당 파일을 생성할 예정이다.
add_message_files(FILES MsgTutorial.msg)

## 의존하는 메세지를 설정하는 옵션이다.
## std_msgs가 설치되어 있지 않다면 빌드 도중에 에러가 난다.
## 또 다른 메세지 패키지를 추가할수도 있다.
generate_messages(DEPENDENCIES std_msgs)

## catkin 패키지 옵션으로 라이브러리, catkin 빌드 의존성, 시스템 의존 패키지를 기술한다.
catkin_package(
LIBRARIES ros_tutorials_topic
CATKIN_DEPENDS std_msgs roscpp
)

## Include 디렉토리를 설정한다.
## catkin_INCLUDE_DIRS는 기본적으로 이미 존재하는 디렉토리이고, 패키지를 만들면서 패키지 include 폴더 내에 헤더파일이 더 추가될 경우, ${catkin_INCLUDE_DIRS} 뒤쪽에 include를 추가해주면 된다.
include_directories(${catkin_INCLUDE_DIRS})

## topic_publisher 노드에 대한 빌드 옵션이다.
## 이름에서도 알 수 있듯이 topic을 송신할 노드를 생성하는 명령어이다.
## add_executable 명령어에 topic_publisher라고 작성하면, 해당 명으로 실행 파일이 생성된다.
## src/topic_publisher.cpp은 실행 파일 생성 시 사용하는 소스 코드이다.
## add_dependencies는 추가 의존성을 설정하는 명령어이고, target_link_libraries는 타겟 링크 라이브러리를 설정하는 명령어이다.
add_executable(topic_publisher src/topic_publisher.cpp)
add_dependencies(topic_publisher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(topic_publisher ${catkin_LIBRARIES})

## topic_subscriber 노드에 대한 빌드 옵션이다.
## Topic을 수신할 노드를 생성한다.
## 기본적인 구조는 topic_publisher의 내용과 동일하다.
add_executable(topic_subscriber src/topic_subscriber.cpp)
add_dependencies(topic_subscriber ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(topic_subscriber ${catkin_LIBRARIES})
```
- 위와 같이 빌드 설정 파일을 작성한 후 창을 닫는다.

#### 4.메세지 파일 작성
```
$ roscd ros_tutorials_topic
$ mkdir msg
$ cd msg
$ gedit MsgTutorial.msg
```
- `roscd` 명령어는 기본적으로 `cd` 명령어와 동일하지만, ROS 패키지 명을 입력했을 때 해당 폴더로 바로 이동하기 때문에 ROS를 사용할 때 편리하다.
- 앞서 `CMakeLists.txt` 파일에 `add_message_files(FILES MsgTutorial.msg)`라는 옵션을 넣었고 이에 관한 파일을 생성하는 내용이다.
- 패키지 폴더 내에 `msg`라는 폴더를 생성하고 `MsgTutorial`이라는 메세지 명으로 메세지 파일을 작성한다.
```
time stamp
int32 data
```
- `MsgTutorial.msg`에 위와 같이 메세지 타입과 메세지 변수명으로 작성한다.
- `time`은 현재 시각을 입력하기 위한 변수이다.
- 메세지 타입은 `bool, int8, int16, float32, string, time, duration` 등의 기본 타입과 ROS에서 많이 사용되는 메세지를 모아놓은 `common_msgs`도 있다.
- 메세지 타입에 대해서는 아래의 링크를 참고하면 된다.  
[message_type](http://wiki.ros.org/msg)
- 메세지 파일의 경우 빌드를 하면 자동으로 헤더파일(위의 경우 `MsgTutorial.h`)을 생성해준다고 한다.

#### 5. publisher 노드 작성
```
$ roscd ros_tutorials_topic/src
$ gedit topic_publisher.cpp
```
- 앞서 `CMakeLists.txt` 파일에 `add_executable(topic_publisher src/topic_publisher.cpp)`라는 옵션을 넣었고 이에 관한 파일을 생성하는 내용이다.
- src 폴더의 `topic_publisher.cpp`를 빌드하여 `topic_publisher`라는 실행 파일을 생성하게 된다.
``` cpp
#include "ros/ros.h"                         // ROS 기본 헤더파일로, ROS 관련 API가 포함된다.
#include "ros_tutorials_topic/MsgTutorial.h" // MsgTutorial 메세지 파일 헤더(빌드 후 자동 생성됨)
int main(int argc, char **argv) // 노드 main 함수
{
  // 아래의 2줄은 초기화 작업으로, 필수적인 부분이다.
  ros::init(argc, argv, "topic_publisher"); // 노드명 초기화
  ros::NodeHandle nh;                       // ROS 시스템과 통신을 위한 노드 핸들 선언

  // publisher 선언, ros_tutorials_topic 패키지의 MsgTutorial 메세지 파일을 이용한
  // publisher ros_tutorial_pub 를 작성한다. 토픽명은 "ros_tutorial_msg" 이며,
  // publisher 큐(queue) 사이즈를 100개로 설정한다는 것이다.
  // publisher 큐 사이즈를 100으로 한다는 것은 메세지를 동시에 100개 보낼 수 있다는 의미인데, 예제를 실행할 때에는 1로 해도 무방하고 디스크의 용량이 클 경우 넉넉히 할당해도 된다고 한다. 
  ros::Publisher ros_tutorial_pub = nh.advertise<ros_tutorials_topic::MsgTutorial>("ros_tutorial_msg", 100);

  // 루프 주기를 설정한다. "10" 이라는 것은 10Hz를 말하는 것으로 0.1초 간격으로 반복된다.
  ros::Rate loop_rate(10);

  // MsgTutorial 메세지 파일 형식으로 msg 라는 메세지를 선언.
  ros_tutorials_topic::MsgTutorial msg;

  // 메세지에 사용될 변수 선언.
  int count = 0;

  // ros::ok()는 c에서 1(true)와 비슷한 함수이고 종료할 때에는 ctrl + c 를 사용하면 된다.
  while (ros::ok())
  {
    msg.stamp = ros::Time::now(); // 현재 시간을 msg의 하위 stamp 메세지에 담는다.
    msg.data = count;             // count라는 변수 값을 msg의 하위 data 메세지에 담는다.

    // ROS_INFO는 c에서 printf와 동일한 것이라고 보면 된다.
    // ROS에서는 printf를 ROS_INFO, ROS_WARN, ROS_ERROR 등 세부적으로 나누어 놓았다고 한다.
    ROS_INFO("send msg = %d", msg.stamp.sec);  // stamp.sec 메세지를 표시한다.
    ROS_INFO("send msg = %d", msg.stamp.nsec); // stamp.nsec 메세지를 표시한다.
    ROS_INFO("send msg = %d", msg.data);       // data 메세지를 표시한다.
    
    ros_tutorial_pub.publish(msg); // publish 멤버 함수를 통해 메세지를 publish한다.
    loop_rate.sleep(); // 위에서 정한 루프 주기에 따라 슬립에 들어간다.
    ++count;           // count 변수 1씩 증가.
  }
  return 0;
}
```
- 노드 및 topic 정보를 요약하면 아래와 같다.
  > 노드 명: topic_publisher  
    topic 명: ros_tutorial_msg  
    topic 타입:  ros_tutorials_topic::MsgTutorial

#### 6. subscriber 노드 작성
```
$ roscd ros_tutorials_topic/src
$ gedit topic_subscriber.cpp
```
- 앞서 `CMakeLists.txt` 파일에 `add_executable(topic_subscriber src/topic_subscriber.cpp)`라는 옵션을 넣었고 이에 관한 파일을 생성하는 내용이다.
- src 폴더의 `topic_subscriber.cpp`를 빌드하여 `topic_subscriber`라는 실행 파일을 생성하게 된다.
``` cpp
#include "ros/ros.h"                         // ROS 기본 헤더파일
#include "ros_tutorials_topic/MsgTutorial.h" // MsgTutorial 메세지 파일 헤더(빌드 후 자동 생성됨)

// 메세지 콜백 함수로써, 밑에서 설정한 ros_tutorial_msg라는 이름의 topic
// 메세지를 수신하였을 때 동작하는 함수이다.
// 입력 변수로 ros_tutorials_topic 패키지의 MsgTutorial 메세지를 받도록 되어 있다.
void msgCallback(const ros_tutorials_topic::MsgTutorial::ConstPtr& msg)
{
  ROS_INFO("recieve msg = %d", msg->stamp.sec);  // stamp.sec 메세지를 표시한다.
  ROS_INFO("recieve msg = %d", msg->stamp.nsec); // stamp.nsec 메세지를 표시한다.
  ROS_INFO("recieve msg = %d", msg->data);       // data 메세지를 표시한다.
}
int main(int argc, char **argv) // 노드 main 함수
{
ros::init(argc, argv, "topic_subscriber"); // 노드명 초기화
ros::NodeHandle nh; // ROS 시스템과 통신을 위한 노드 핸들 선언

// subscriber 선언, ros_tutorials_topic 패키지의 MsgTutorial 메세지 파일을 이용한
// subscriber ros_tutorial_sub 를 작성한다. 토픽명은 "ros_tutorial_msg" 이며,
// subscriber 큐(queue) 사이즈를 100개로 설정한다는 것이다.
// topic이 수신되면 subscriber는 msgCallback 함수를 실행한다.
ros::Subscriber ros_tutorial_sub = nh.subscribe("ros_tutorial_msg", 100, msgCallback);

// 콜백함수 호출을 위한 함수로써, 메세지가 수신되기를 대기,
// 수신되었을 경우 콜백함수를 실행한다.
ros::spin();

return 0;
}
```
- 노드 및 topic 정보를 요약하면 아래와 같다.
  > 노드 명: topic_subscriber  
    topic 명: ros_tutorial_msg  
    topic 타입: ros_tutorials_topic::MsgTutorial
- ROS에서 통신을 할 때에는 위의 topic 명과 topic 타입을 비교하여 publisher가 subscriber에게 topic을 보낸다.
- 따라서 위와 같이 publisher와 subscriber의 topic 명 및 topic 타입은 동일해야 한다.

#### 7. ROS 노드 빌드
```
$ cd ~/catkin_ws
$ catkin_make
```
- 패키지 작성이 완료되면, `catkin_make` 명령어를 통해 패키지의 메세지 파일, publisher 노드, subscriber 노드를 빌드한다.
- 빌드된 결과물은 `/catkin_ws`의 `/build`와 `/devel` 폴더에 각각 아래와 같이 생성된다.
  - `/build` 폴더에는 catkin 빌드에서 사용된 설정 내용이 저장
  - `/devel/lib/ros_tutorials_topic` 폴더에는 실행 파일이 저장
  - `/devel/include/ros_tutorials_topic` 폴더에는 메시지 파일로부터 자동 생성된 메세지 헤더 파일이 저장

#### 8. publisher 실행
```
$ rosrun ros_tutorials_topic topic_publisher
```
- 노드 실행에 앞서 `roscore를 반드시 먼저 실행한다.
- ROS 노드 실행 명령어인 `rosrun`을 이용하여 `ros_tutorials_topic` 패키지의 `topic_publisher` 노드를 구동하라는 명령이다.

```
$ rostopic list
/ros_tutorial_msg
/rosout
/rosout_agg

$ rostopic info /ros_tutorial_msg
$ rostopic echo /ros_tutorial_msg
```
- `rostopic` 명령어는 현재 ROS 네트워크에서 사용 중인 topic의 목록, 주기, 데이터 대역폭, 내용 확인 등을 할 수 있는 명령어이다.

#### 9. subscriber 실행
```
$ rosrun ros_tutorials_topic topic_subscriber
```
- ROS 노드 실행 명령어인 `rosrun`을 이용하여 `ros_tutorials_topic` 패키지의 `topic_subscriber` 노드를 구동하라는 명령이다.
- subscriber 노드가 실행되면 `ROS_INFO`를 통해 출력한 내용들이 터미널 창에 보인다.

#### 10. 실행된 노드들의 통신 상태 확인
```
$ rqt_graph
```
- 지금까지 ROS 패키지 생성하는 법과 topic을 송/수신하는 법에 대해서 살펴보았다.
- 다음으로는 sevice를 송/수신하는 패키지를 생성하고 송/수신하는 법에 대해 정리해볼 예정인데, 내용은 topic 패키지와 유사하다.

---
- 아래는 깃허브에서 위와 동일한 소스 코드를 가져와 노드를 실행하는 방법을 정리한 것이다.
- `catkin_ws/src` 폴더에 아래의 명령어로 소스 코드를 클론한 후 빌드하고, 실행 파일들을 차례로 실행하면 된다.
- 아래의 내용은 추후의 다른 ROS 공식 패키지들을 사용할 때에도 필요한 내용이므로 정리해두었다.
```
$ cd ~/catkin_ws/src
$ git clone https://github.com/ROBOTIS-GIT/ros_tutorials.git
$ cd ~/catkin_ws
$ catkin_make
```
```
$ rosrun ros_tutorials_topic topic_publisher
```
```
$ rosrun ros_tutorials_topic topic_subscriber
```

# 2021/04/11
### service 송수신
- 위에서 살펴본 `topic`은 단방향 메세지 통신인 반면 `service`는 양방향 메세지 통신 방법이다.
- `service`에서는 요청(request)이 있을 때에만 응답(respond)하는 `서비스 서버(service server)`와 요청한 후 응답을 받는 `서비스 클라이언트(service client)`로 나뉜다.
- 아래에서는 `service`를 송수신하는 패키지를 생성한 후 `service`를 직접 송수신하는 실습을 진행해보려 한다.

#### 1. 패키지 생성
```
$ cd ~/catkin_ws/src
$ catkin_create_pkg ros_tutorials_service message_generation std_msgs roscpp
$ cd ros_tutorials_service
$ ls
include        -> 헤더 파일 폴더
src            -> 소스 코드 폴더
CMakeLists.txt -> 빌드 설정 파일
package.xml    -> 패키지 설정 파일
```

#### 2. 패키지 설정 파일(package.xml) 수정
- 패키지 설정 파일의 구성은 topic 패키지의 패키지 설정 파일 구성과 유사하므로 자세한 설명은 생략한다.
```
$ gedit package.xml
```
``` xml
<?xml version="1.0"?>
<package>
<name>ros_tutorials_service</name>
<version>0.1.0</version>
<description>ROS tutorial package to learn the service</description>
<license>Apache License 2.0</license>
<author email="pyo@robotis.com">Yoonseok Pyo</author>
<maintainer email="pyo@robotis.com">Yoonseok Pyo</maintainer>
<url type="bugtracker">https://github.com/ROBOTIS-GIT/ros_tutorials/issues</url>
<url type="repository">https://github.com/ROBOTIS-GIT/ros_tutorials.git</url>
<url type="website">http://www.robotis.com</url>
<buildtool_depend>catkin</buildtool_depend>
<build_depend>roscpp</build_depend>
<build_depend>std_msgs</build_depend>
<build_depend>message_generation</build_depend>
<run_depend>roscpp</run_depend>
<run_depend>std_msgs</run_depend>
<run_depend>message_runtime</run_depend>
<export></export>
</package>
```
#### 3. 빌드 설정 파일(CMakeLists.txt) 수정
- 빌드 설정 파일의 구성 또한 topic 패키지의 빌드 설정 파일의 구성과 유사하다.
```
$ gedit CMakeLists.txt
```
``` cmake
cmake_minimum_required(VERSION 2.8.3)
project(ros_tutorials_service)

## 캐킨 빌드를 할 때 요구되는 구성요소 패키지이다.
## 의존성 패키지로 message_generation, std_msgs, roscpp이며 이 패키지들이 존재하지 않으면 빌드 도중에 에러가 난다.
find_package(catkin REQUIRED COMPONENTS message_generation std_msgs roscpp)

## 서비스 선언: SrvTutorial.srv
add_service_files(FILES SrvTutorial.srv)

## 의존하는 메시지를 설정하는 옵션이다.
## std_msgs가 설치되어 있지 않다면 빌드 도중에 에러가 난다.
generate_messages(DEPENDENCIES std_msgs)

## 캐킨 패키지 옵션으로 라이브러리, 캐킨 빌드 의존성, 시스템 의존 패키지를 기술한다.
catkin_package(
LIBRARIES ros_tutorials_service
CATKIN_DEPENDS std_msgs roscpp
)

## 인클루드 디렉터리를 설정한다.
include_directories(${catkin_INCLUDE_DIRS})

## service_server 노드에 대한 빌드 옵션이다.
## 실행 파일, 타겟 링크 라이브러리, 추가 의존성 등을 설정한다.
add_executable(service_server src/service_server.cpp)
add_dependencies(service_server ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(service_server ${catkin_LIBRARIES})

## service_client 노드에 대한 빌드 옵션이다.
add_executable(service_client src/service_client.cpp)
add_dependencies(service_client ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(service_client ${catkin_LIBRARIES})
```

#### 4. 서비스 파일 작성
```
$ roscd ros_tutorials_service -> 패키지 폴더로 이동
$ mkdir srv
$ cd srv
$ gedit SrvTutorial.srv       -> SrvTutorial.srv 파일 신규 작성 및 내용 수정
```
- 앞서 `CMakeLists.txt` 파일에 `add_service_files(FILES SrvTutorial.srv)`라는 옵션을 넣었고 이에 관한 파일을 생성하는 내용이다.
- 패키지 폴더 내에 `srv`라는 폴더를 생성하고 `SrvTutorial`이라는 메세지 명으로 메세지 파일을 작성한다.
```
int64 a
int64 b
---
int64 result
```
- `SrvTutorial.srv`에 위와 같이 메세지 타입과 메세지 변수명으로 작성한다.
- 여기서 `---`는 요청과 응답을 구분하는 구분자로, `---` 위쪽의 변수는 서비스 클라이언트가 서비스 서버에 서비스 요청을 할 때 전달하는 변수이고, `---` 아래쪽의 변수는 서비스 서버가 서비스 클라이언트에게 서비스를 응답할 때 전달하는 변수이다.

#### 5. Service server 노드 작성
```
$ roscd ros_tutorials_service/src
$ gedit service_server.cpp
```
- 앞서 `CMakeLists.txt` 파일에 `add_executable(service_server src/service_server.cpp)`라는 옵션을 넣었고 이에 관한 파일을 생성하는 내용이다.
- src 폴더의 `service_server.cpp`를 빌드하여 `service_server`라는 실행 파일을 생성하게 된다.
``` cpp
#include "ros/ros.h"                           // ROS 기본 헤더 파일
#include "ros_tutorials_service/SrvTutorial.h" // SrvTutorial 서비스 파일 헤더 (빌드후 자동 생성됨)

// 서비스 요청이 있을 경우, 아래의 처리를 수행한다.
// 서비스 요청은 req, 서비스 응답은 res로 설정하였다.
bool calculation(ros_tutorials_service::SrvTutorial::Request &req,
ros_tutorials_service::SrvTutorial::Response &res)
{
  // 서비스 요청시 받은 a와 b 값을 더하여 서비스 응답 값에 저장한다.
  res.result = req.a + req.b;

  // 서비스 요청에 사용된 a, b 값의 표시 및 서비스 응답에 해당되는 result 값을 출력한다.
  ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
  ROS_INFO("sending back response: %ld", (long int)res.result);

  return true;
}

int main(int argc, char **argv) // 노드 main 함수
{
  ros::init(argc, argv, "service_server"); // 노드명 초기화
  ros::NodeHandle nh;                      // 노드 핸들 선언

  // 서비스 서버 선언, ros_tutorials_service 패키지의 SrvTutorial 서비스 파일을 이용한
  // 서비스 서버 ros_tutorials_service_server를 선언한다.
  // 서비스명은 ros_tutorial_srv이며 서비스 요청이 있을 때,
  // calculation라는 함수를 실행하라는 설정이다.
  ros::ServiceServer ros_tutorials_service_server = nh.advertiseService("ros_tutorial_srv", calculation);
  ROS_INFO("ready srv server!");

  ros::spin(); // 서비스 요청을 대기한다.

  return 0;
}
```
- 노드 및 service 정보를 요약하면 아래와 같다.
  > 노드 명: service_server  
    service 명: ros_tutorial_srv  
    service 타입: ros_tutorials_service::SrvTutorial

#### 6. Service client 노드 작성
```
$ roscd ros_tutorials_service/src
$ gedit service_client.cpp
```
- 앞서 `CMakeLists.txt` 파일에 `add_executable(service_client src/service_client.cpp)`라는 옵션을 넣었고 이에 관한 파일을 생성하는 내용이다.
- src 폴더의 `service_client.cpp`를 빌드하여 `service_client`라는 실행 파일을 생성하게 된다.
``` cpp
#include "ros/ros.h"                           // ROS 기본 헤더 파일
#include "ros_tutorials_service/SrvTutorial.h" // SrvTutorial 서비스 파일 헤더 (빌드후 자동 생성됨)
#include <cstdlib>                             // atoll 함수 사용을 위한 라이브러리

int main(int argc, char **argv) // 노드 main 함수
{
  ros::init(argc, argv, "service_client"); // 노드명 초기화
  
  if (argc != 3)                           // 입력값 오류 처리
  {
    ROS_INFO("cmd : rosrun ros_tutorials_service service_client arg0 arg1");
    ROS_INFO("arg0: double number, arg1: double number");
    return 1;
  }

  ros::NodeHandle nh; // ROS 시스템과 통신을 위한 노드 핸들 선언

  // 서비스 클라이언트 선언, ros_tutorials_service 패키지의 SrvTutorial 서비스 파일을 이용한
  // 서비스 클라이언트 ros_tutorials_service_client를 선언한다.
  // 서비스명은 "ros_tutorial_srv"이다.
  ros::ServiceClient ros_tutorials_service_client =
  nh.serviceClient<ros_tutorials_service::SrvTutorial>("ros_tutorial_srv");

  // srv라는 이름으로 SrvTutorial 서비스 파일을 이용하는 서비스를 선언한다.
  ros_tutorials_service::SrvTutorial srv;

  // 서비스 요청 값으로 노드가 실행될 때 입력으로 사용된 매개변수를 각각의 a, b에 저장한다.
  srv.request.a = atoll(argv[1]); // atoll()은 string to int 함수이다.
  srv.request.b = atoll(argv[2]);

  // call() 멤버 함수를 통해 서비스를 요청하고, 요청이 받아들여졌을 경우, 응답 값을 표시한다.
  if (ros_tutorials_service_client.call(srv))
  {
    ROS_INFO("send srv, srv.Request.a and b: %ld, %ld", (long int)srv.request.a, (long int)srv.request.b);
    ROS_INFO("receive srv, srv.Response.result: %ld", (long int)srv.response.result);
  }
  else
  {
    ROS_ERROR("Failed to call service ros_tutorial_srv");
    return 1;
  }
  return 0;
}
```
- 노드 및 service 정보를 요약하면 아래와 같다.
  > 노드 명: service_client
    service 명: ros_tutorial_srv  
    service 타입: ros_tutorials_service::SrvTutorial
- ROS에서 통신을 할 때에는 위의 service 명과 service 타입을 비교하여 service server가 service client에게 service을 보낸다.
- 따라서 위와 같이 service server와 service client의 service 명 및 service 타입은 동일해야 한다.

#### 7. ROS 노드 빌드
```
$ cd ~/catkin_ws && catkin_make
```
- 패키지 작성이 완료되면, `catkin_make` 명령어를 통해 패키지의 메세지 파일, service server 노드, service client 노드를 빌드한다.
- 빌드된 결과물은 `/catkin_ws`의 `/build`와 `/devel` 폴더에 각각 아래와 같이 생성된다.
  - `/build` 폴더에는 catkin 빌드에서 사용된 설정 내용이 저장
  - `/devel/lib/ros_tutorials_service` 폴더에는 실행 파일이 저장
  - `/devel/include/ros_tutorials_service` 폴더에는 메시지 파일로부터 자동 생성된 메세지 헤더 파일이 저장

#### 8. Service server 실행
```
$ rosrun ros_tutorials_service service server
```
- 노드 실행에 앞서 `roscore`를 반드시 먼저 실행한다.
- ROS 노드 실행 명령어인 `rosrun`을 이용하여 `ros_tutorials_service` 패키지의 `service server` 노드를 구동하라는 명령이다.
- `Service server` 노드는 서비스 요청이 있기 전까지 아무런 처리를 하지 않고 대기 상태에서 요청을 기다린다.
#### 9. Service client 실행
```
$ rosrun ros_tutorials_service service_client 2 3
[INFO] [1495726543.277216401]: send srv, srv.Request.a and b: 2, 3
[INFO] [1495726543.277258018]: receive srv, srv.Response.result: 5
```
- ROS 노드 실행 명령어인 `rosrun`을 이용하여 `ros_tutorials_service` 패키지의 `service_client` 노드를 구동하라는 명령이다.
- 노드를 실행시킬 때 요청할 때 사용할 실행 매개변수에 입력할 값(2, 3)도 함께 입력한다.
- 2와 3은 각각 a, b 값으로 서비스를 요청하고, 이에 대한 결과로 둘의 합인 5가 result에 입력되어 응답 값으로 전송받은 것을 보여준다.
- 실제 활용에서는 이러한 실행 매개변수 대신 명령어나 계산되어야 할 값, 트리거용 변수 등을 서비스 요청 값으로 사용할 수도 있다.

```
$ rosservice call /ros_tutorial_srv 10 2
```
- 서비스 요청을 할 때에는 service client 노드를 실행할 수도 있지만, `rosservice call` 명령어를 사용할 수도 있다.
```
$ rqt
```
- ROS의 GUI 도구인 rqt를 사용하는 방법도 있다.
- rqt에서 [플러그인(Plugins)] → [서비스(Service)] →
[Service Caller]를 선택한 후, service 칸에서 /ros_tutorial_srv를 선택하고 a, b에 값을 입력, call을 클릭하면 result에 값이 표시된다.
---
- 지금까지 양방향 메세지 통신 `service`를 송/수신하는 법에 대해서 살펴보았다.
- 다음으로는 `roscore`의 역할 중 하나인 `parameter server`를 활용하는 방법과 `roslaunch`를 사용하는 방법을 정리해볼 예정이다.

# 2021/04/21
- 실차에 환경을 구축하면서 ROS를 사용한 서버-실차 간 통신이 원할하게 이루어지지 않아 ROS는 사용하지 않게 되었다.
- 그러면서 ROS 공부도 지지부진해져서 정리도 늦어지게 되었지만, 더 늦으면 또 마무리를 못할 것 같아서 공부한 부분까지는 정리해놓으려고 한다.
### Parameter
- `roscore`의 역할 중 하나는 파라미터 서버를 관리하는 것이다.
- 파라미터는 토픽이나 서비스를 송수신할 때 사용하는 변수 외에 사용자가 노드 외부로부터 노드의 흐름이나 설정, 처리 등을 바꿀 수 있는 매우 유용한 기능이다.
- 위에서 작성한 ros_tutorials_service.cpp에 파라미터를 추가하여 실습해보도록 한다.
#### 1. 파라미터를 활용한 노드 작성
```
$ roscd ros_tutorials_service/src
$ gedit service_server.cpp
```
- 파라미터 관련 설정 `setParam` 및 읽기 `getParam` 함수에 주목하여 코드를 보면 된다.
- 파라미터는 int, float, bool, string, dictionary, list 등 다양한 타입으로 설정할 수 있다.
``` cpp
#include "ros/ros.h"                           // ROS 기본 헤더파일
#include "ros_tutorials_service/SrvTutorial.h" // SrvTutorial 서비스 파일 헤더 (빌드 후 자동 생성됨)
#define PLUS 1           // 덧셈
#define MINUS 2          // 빼기
#define MULTIPLICATION 3 // 곱하기
#define DIVISION 4       // 나누기

// 기존의 서비스 서버 코드와 다른 점은 g_operator라는 변수가 추가되었다는 점이다.
// 여기서 g_operator가 파라미터 역할을 하는 변수로 디폴트 값은 PLUS이지만, 
// 서비스 서버 노드 실행 시 사용자가 해당 값을 변경시켜가며 노드를 실행할 수 있다.
int g_operator = PLUS;

// 서비스 요청이 있을 경우, 아래의 처리를 수행한다.
// 서비스 요청은 req, 서비스 응답은 res로 설정하였다.
bool calculation(ros_tutorials_service::SrvTutorial::Request &req,
                 ros_tutorials_service::SrvTutorial::Response &res)
{
// 서비스 요청시 받은 a와 b 값을 파라미터 값에 따라 연산자를 달리한다.
// 계산한 후 서비스 응답 값에 저장한다.
  switch(g_operator)
  {
    case PLUS:
      res.result = req.a + req.b; break;
    case MINUS:
      res.result = req.a - req.b; break;
    case MULTIPLICATION:
      res.result = req.a * req.b; break;
    case DIVISION:
      if(req.b == 0){
        res.result = 0; break;
      }
      else{
        res.result = req.a / req.b; break;
      }
    default:
      res.result = req.a + req.b; break;
  }

  // 서비스 요청에 사용된 a, b값의 표시 및 서비스 응답에 해당되는 result 값을 출력한다.
  ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
  ROS_INFO("sending back response: [%ld]", (long int)res.result);

  return true;
}

int main(int argc, char **argv)            // 노드 메인 함수
{
  ros::init(argc, argv, "service_server"); // 노드명 초기화

  ros::NodeHandle nh;                      // ROS 시스템과 통신을 위한 노드 핸들 선언

  nh.setParam("calculation_method", PLUS); // 파라미터 초기설정

  // 서비스 서버 선언, ros_tutorials_service 패키지의 SrvTutorial 서비스 파일을 이용한
  // 서비스 서버 service_server를 작성한다. 서비스명은 "ros_tutorial_srv"이며,
  // 서비스 요청이 있을 때, calculation라는 함수를 실행하라는 설정이다.
  ros::ServiceServer ros_tutorial_service_server = nh.advertiseService("ros_tutorial_srv", calculation);

  ROS_INFO("ready srv server!");
  ros::Rate r(10); // 10 hz

  while (1)
  {
    nh.getParam("calculation_method", g_operator); // 연산자를 매개변수로부터 받은 값으로 변경한다.
    
    // 기존의 서비스를 송신할 때에는 ros::spin();을 사용하여 클라이언트 서버의 요청을 대기하였지만,
    // 파라미터를 사용할 때에는 서비스 요청 응답 후 파라미터를 다시 받으므로 spinOnce()를 사용하여 한번 요청에 응답하면
    // 다음 코드로 넘어가도록 한다.
    ros::spinOnce();                               // 콜백함수 처리루틴
    r.sleep();                                     // 루틴 반복을 위한 sleep 처리
  }

  return 0;
}
```
- 노드 및 service 정보를 요약하면 아래와 같다.
  > 노드 명: service_server  
    service 명: ros_tutorial_srv  
    service 타입: ros_tutorials_service::SrvTutorial

#### 2. 노드 빌드 및 실행
```
$ cd ~/catkin_ws && catkin_make
$ rosrun ros_tutorials_service service_server
[INFO] [1495767130.149512649]: ready srv server! # ROSINFO 출력이 보이면 노드가 제대로 실행된 것이다.
```
#### 3. 파라미터 목록 보기
- `rosparam list` 명렬어를 사용하면 현재 ROS 네트워크에 사용된 파라미터의 목록을 확인할 수 있다.
- /calculation_method가 현재 ROS 서비스 서버가 생성한 파라미터이다.
```
$ rosparam list
/calculation_method
/rosdistro
/rosversion
/run_id
```
#### 4. 파라미터 사용 예
``` 
$ rosservice call /ros_tutorial_srv 10 5 # 사칙연산의 변수 a, b 입력
result: 15                               # 디폴트 사칙연산인 덧셈 결괏값

$ rosparam set /calculation_method 2     # MINUS로 파라미터 변경
$ rosservice call /ros_tutorial_srv 10 5
result: 5

$ rosparam set /calculation_method 3     # MULTIPLICATION으로 파라미터 변경
$ rosservice call /ros_tutorial_srv 10 5
result: 50

$ rosparam set /calculation_method 4     # DEVISION으로 파라미터 변경
$ rosservice call /ros_tutorial_srv 10 5
result: 2
```