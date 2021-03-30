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