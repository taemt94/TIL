#### 2022/03/27
- [ROS2로 시작하는 로봇 프로그래밍](https://cafe.naver.com/openrt/24070)을 참조하여 ROS2를 학습하고, 필요한 코드들을 정리해놓았다.
# 3. ROS2 개발 환경 구축
- [데비안 패키지를 이용한 ROS2 설치](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)

## 3.4. ROS2 개발 툴 설치
``` bash
sudo apt update && sudo apt install -y \
  build-essential \
  cmake \
  git \
  libbullet-dev \
  python3-colcon-common-extensions \
  python3-flake8 \
  python3-pip \
  python3-pytest-cov \
  python3-rosdep \
  python3-setuptools \
  python3-vcstool \
  wget
```
``` bash
python3 -m pip install -U \
  argcomplete \
  flake8-blind-except \
  flake8-builtins \
  flake8-class-newline \
  flake8-comprehensions \
  flake8-deprecated \
  flake8-docstrings \
  flake8-import-order \
  flake8-quotes \
  pytest-repeat \
  pytest-rerunfailures \
  pytest
```
``` bash
sudo apt install --no-install-recommends -y \
  libasio-dev \
  libtinyxml2-dev \
  libcunit1-dev
```
## 3.5. ROS2 빌드 테스트

``` bash
$ source /opt/ros/foxy/setup.bash # Set up environment by sourcing setup file
$ mkdir -p ~/robot_ws/src         # 워크스페이스 폴더 생성
$ cd ~/robot_ws/
$ colcon build --symlink-install  # 빌드
```

## 3.6. Run commands 설정
- 필요한 명령어만 bashrc에 설정하여 사용
``` bash
$ vim ~/.bashrc
```
```
source /opt/ros/foxy/setup.bash
source ~/robot_ws/install/local_setup.bash

source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
source /usr/share/vcstool-completion/vcs.bash
source /usr/share/colcon_cd/function/colcon_cd.sh
export _colcon_cd_root=~/robot_ws

export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=robot1

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# export RMW_IMPLEMENTATION=rmw_connext_cpp
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# export RMW_IMPLEMENTATION=rmw_gurumdds_cpp

# export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity} {time}] [{name}]: {message} ({function_name}() at {file_name}:{line_number})'
export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}]: {message}'
export RCUTILS_COLORIZED_OUTPUT=1
export RCUTILS_LOGGING_USE_STDOUT=0
export RCUTILS_LOGGING_BUFFERED_STREAM=1

alias cw='cd ~/robot_ws'
alias cs='cd ~/robot_ws/src'
alias ccd='colcon_cd'

alias cb='cd ~/robot_ws && colcon build --symlink-install'
alias cbs='colcon build --symlink-install'
alias cbp='colcon build --symlink-install --packages-select'
alias cbu='colcon build --symlink-install --packages-up-to'
alias ct='colcon test'
alias ctp='colcon test --packages-select'
alias ctr='colcon test-result'

alias rt='ros2 topic list'
alias re='ros2 topic echo'
alias rn='ros2 node list'

alias killgazebo='killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient'

alias af='ament_flake8'
alias ac='ament_cpplint'

alias testpub='ros2 run demo_nodes_cpp talker'
alias testsub='ros2 run demo_nodes_cpp listener'
alias testpubimg='ros2 run image_tools cam2image'
alias testsubimg='ros2 run image_tools showimage'
```

## 3.7 통합 개발환경(IDE) 설치
### 3.7.1 Visual Studio Code
- User settings 설정
  ``` bash
  $ vim ~/.config/Code/User/settings.json
  ```
  ``` bash
  {
    "cmake.configureOnOpen": false,
    "editor.minimap.enabled": false,
    "editor.mouseWheelZoom": true,
    "editor.renderControlCharacters": true,
    "editor.rulers": [100],
    "editor.tabSize": 2,
    "files.associations": {
      "*.repos": "yaml",
      "*.world": "xml",
      "*.xacro": "xml"
    },
    "files.insertFinalNewline": true,
    "files.trimTrailingWhitespace": true,
    "terminal.integrated.scrollback": 1000000,
    "workbench.iconTheme": "vscode-icons",
    "workbench.editor.pinnedTabSizing": "compact",
    "ros.distro": "foxy",
    "colcon.provideTasks": true
  }
  ```

- C/C++ properties 설정
  ``` bash
  $ vim ~/{Your Workspace}/.vscode/c_cpp_properties.json
  ```
  ``` bash
  {
    "configurations": [
      {
        "name": "Linux",
        "includePath": [
          "${default}",
          "${workspaceFolder}/**",
          "/opt/ros/foxy/include/**"
        ],
        "defines": [],
        "compilerPath": "/usr/bin/g++",
        "cStandard": "c99",
        "cppStandard": "c++14",
        "intelliSenseMode": "linux-gcc-x64"
      }
    ],
    "version": 4
  }
  ```
- Tasks 설정
  ``` bash
  $ vim ~/{Your Workspace}/.vscode/tasks.json
  ```
  ``` bash
  {
    "version": "2.0.0",
    "tasks": [
      {
        "label": "colcon: build",
        "type": "shell",
        "command": "colcon build --cmake-args '-DCMAKE_BUILD_TYPE=Debug'",
        "problemMatcher": [],
        "group": {
          "kind": "build",
          "isDefault": true
        }
      },
      {
        "label": "colcon: test",
        "type": "shell",
        "command": "colcon test && colcon test-result"
      },
      {
        "label": "colcon: clean",
        "type": "shell",
        "command": "rm -rf build install log"

      }
    ]
  }
  ```
- Launch 설정
  ``` bash
  $ vim ~/{Your Workspace}/.vscode/launch.json
  ```
  ``` bash
  {
    "version": "0.2.0",
    "configurations": [
      {
        "name": "Debug-rclpy(debugpy)",
        "type": "python",
        "request": "launch",
        "program": "${file}",
        "console": "integratedTerminal"
      },
      {
        "name": "Debug-rclcpp(gbd)",
        "type": "cppdbg",
        "request": "launch",
        "program": "${workspaceFolder}/install/${input:package}/lib/${input:package}/${input:node}",
        "args": [],
        "preLaunchTask": "colcon: build",
        "stopAtEntry": true,
        "cwd": "${workspaceFolder}",
        "externalConsole": false,
        "MIMode": "gdb",
        "setupCommands": [
          {
            "description": "Enable pretty-printing for gdb",
            "text": "-enable-pretty-printing",
            "ignoreFailures": true
          }
        ]
      }
    ],
    "inputs": [
      {
        "id": "package",
        "type": "promptString",
        "description": "package name",
        "default": "topic_service_action_rclcpp_example"
      },
      {
        "id": "node",
        "type": "promptString",
        "description": "node name",
        "default": "argument"
      }
    ]
  }
  ```

### 3.7.2 QtCreator
``` bash
$ sudo apt install qtcreator
$ qtcreator
```

#### 2022/03/28
# 9. 패키지 설치와 노드 실행
## 9.1. Turtlesim 패키지 설치
``` bash
$ sudo apt update
$ sudo apt install ros-foxy-turtlesim
```

## 9.3. Turtlesim 패키지와 노드
- 설치된 패키지 확인
  ``` bash
  $ ros2 pkg list
  ```
- 특정 패키지에 포함된 노드 확인
  ``` bash
  $ ros2 pkg executables <Package Name>
  ```
## 9.4. Turtlesim 패키지의 노드 실행
- 패키지 노드 실행
  ``` bash
  $ ros2 run <Package Name> <Node Executable Name>
  ```
## 9.5. 노드, 토픽, 서비스, 액션의 조회
- 실행 중인 노드 확인
  ``` bash
  $ ros2 node list
  ```
- Topic 확인
  ``` bash
  $ ros2 topic list
  ```
- Service 확인
  ``` bash
  $ ros2 service list
  ```
- Action 확인
  ``` bash
  $ ros2 action list
  ```
## 9.6. rqt_graph로 보는 노드와 토픽의 그래프 뷰
- rqt_graph 실행
  ``` bash
  $ rqt_graph
  ```

# 10. ROS2 노드와 데이터 통신
## 10.2. 노드 실행(ros2 run)
- 노드 실행
  ``` bash
  $ ros2 run <Package Name> <Node Executable Name>
  ```
## 10.3. 노드 목록(ros2 node list)
- 실행 중인 노드 확인
  ``` bash
  $ ros2 node list
  ```
- 노드명 변경하여 노드 실행
  ``` bash
  $ ros2 run <Package Name> <Node Executable Name> __node:=<New Node Name>

## 10.4. 노드 정보(ros2 node info)
- 실행 중인 노드 정보 확인
  $\rightarrow$ 노드의 topic publisher, subscriber, service, action, parameter 등의 정보 확인 가능
  ``` bash
  $ ros2 node info <Node Name>
  ```

# 11. ROS2 토픽
## 11.2. 토픽 목록 확인(ros2 topic list)
- 현재 개발 환경에서 동작 중인 모든 노드들의 토픽 정보 확인
  ``` bash
  $ ros2 topic list -t # -t : 각 메시지의 형태(Type)까지 표시
  ```
- `rqt_graph`에서 `Dead sinks` 및 `Leaf topics` 체크 해제하면 노드 상관없이 모든 토픽 확인 가능

## 11.3. 토픽 정보 확인(ros2 topic info)
- CLI 토픽 정보 확인 명령어
  ``` bash
  $ ros2 topic info <Topic Name>
  ```

## 11.4. 토픽 내용 확인(ros2 topic echo)
- 특정 토픽의 메시지 내용을 실시간으로 표시
  ``` bash
  $ ros2 topic echo <Topic Name>
  ```

## 11.5. 토픽 대역폭 확인(ros2 topic bw)
- 메시지 대역폭(메시지 크기) 확인
  ``` bash
  $ ros2 topic bw <Topic Name>
  ```

## 11.6. 토픽 주기 확인(ros2 topic hz)
- 토픽의 전송 주기 확인
  ``` bash
  $ ros2 topic hz <Topic Name>
  ```

## 11.7. 토픽 지연시간 확인(ros2 topic delay)
- 토픽 지연시간(latency) 확인  
  $\rightarrow$ 메시지 내에 header stamp 메시지를 사용하고 있어야 확인 가능
  ``` bash
  $ ros2 topic delay <Topic Name>
  ```

## 11.8. 토픽 퍼블리시(ros2 topic pub)
- 간단히 하나의 토픽 퍼블리시할 때 사용
  ``` bash
  $ ros2 topic pub <Topic Name> <Message Type> <Message>
  ## Example
  $ ros2 topic pub --once /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.8}}" # --once 옵션: 한 번만 퍼블리시
  $ ros2 topic pub --rate 1 /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.8}}" # --rate 옵션: 뒤에 쓴 주기로 퍼블리시
  ```

## 11.9. bag 기록(ros2 bag record)
- 퍼블리시되는 토픽을 파일 형태로 저장
  ``` bash
  $ ros2 bag record -o <Directory> <Topic Name 1> <Topic Name 2> ... # -o 옵션: rosbag 저장 폴더 명 설정
  ```

## 11.10. bag 정보(ros2 bag info)
- 저장된 rosbag 파일 정보 확인
  ``` bash
  $ ros2 bag info <rosbag Name>
  ```

## 11.11. bag 재생(ros2 bag play)
- 저장된 rosbag 파일을 재생하여 저장한 토픽 확인
  ``` bash
  $ ros2 bag play <rosbag Name>
  $ ros2 topic echo <Topic Name> # bag play 한 후 echo 통해 토픽 확인 가능
  ```

#### 2022/03/31
# 23. ROS2의 빌드 시스템과 빌드 툴
## 23.4. 패키지 생성
- 패키지 생성 명령어
  ``` bash
  $ ros2 pkg create <Package Name> --build-type <Build Type> --dependecies <Dependant Package 1> ... <Dependant Package n>

  ## Example
  $ cd ~/ROS2_ws/src
  $ ros2 pkg create my_first_ros_rclcpp_pkg --build-type ament_cmake --dependencies rclcpp std_msgs ## C++
  $ ros2 pkg create my_first_ros_rclpy_pkg --build-type ament_python --dependencies rclpy std_msgs ## Python
  ```

## 23.5. 패키지 빌드
- colcon 빌드 툴을 사용한 ROS2 패키지 빌드
  ``` bash
  $ cd ~/ROS2_ws
  $ colcon build --symlink-install ## 전체 패키지 모두 빌드
  $ colcon build --symlink-install --packages-select <Package Name> ## 특정 패키지만 빌드
  $ colcon build --symlink-install --packages-up-to <Package Name> ## 특정 패키지 및 해당 패키지의 의존성 패키지까지 함께 빌드
  ```

## 23.6. 빌드 시스템에 필요한 부가 기능
### 23.6.2. rosdep(의존성 관리 툴)
- package.xml에 기술된 의존성 정보를 가지고 의존성 패키지들을 설치해주는 툴
  ``` bash
  $ sudo rosdep init
  $ rosdep update
  $ rosdep install --from-paths src --ignore-src --rosdistro foxy -y --skip-keys "console_bridge fastcdr fastrtps rti-connext-dds-5.3.1 urdfdom_headers"
  ```

# 2부
# 2. ROS 프로그래밍 기초(파이썬)
## 2.3. 패키지 설정
### 2.3.1. 패키지 설정 파일(package.xml)

``` xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_first_ros_rclpy_pkg</name>
  <version>0.0.2</version>
  <description>ROS 2 rclpy basic package for the ROS 2 seminar</description>
  <maintainer email="pyo@robotis.com">Pyo</maintainer>
  <license>Apache License 2.0</license>
  <author email="mikael@osrfoundation.org">Mikael Arguedas</author>
  <author email="pyo@robotis.com">Pyo</author>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type> ## C++ : ament_cmake, python : ament_python
  </export>
</package>
```

### 2.3.2. 파이썬 패키지 설정 파일(setup.py)
- `entry_points` 옵션의 `console_scripts` 키를 사용한 실행 파일 설정 필요
- 예를 들어 `helloworld_publisher`과 `helloworld_subscriber` 콘솔 스크립트는 각각 my_first_ros_rclpy_pkg.helloworld_publisher 모듈과 my_first_ros_rclpy_pkg.helloworld_subscriber 모듈의 main 함수를 호출 $\rightarrow$ `ros2 run` 또는 `ros2 launch`를 이용하여 해당 스크립트 실행 가능
``` python
from setuptools import find_packages
from setuptools import setup

package_name = 'my_first_ros_rclpy_pkg'

setup(
    name=package_name,
    version='0.0.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Mikael Arguedas, Pyo',
    author_email='mikael@osrfoundation.org, pyo@robotis.com',
    maintainer='Pyo',
    maintainer_email='pyo@robotis.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='ROS 2 rclpy basic package for the ROS 2 seminar',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'helloworld_publisher = my_first_ros_rclpy_pkg.helloworld_publisher:main',
            'helloworld_subscriber = my_first_ros_rclpy_pkg.helloworld_subscriber:main',
        ],
    },
)
```

### 2.3.3. 파이썬 패키지 환경설정 파일(setup.cfg)
- colcon 빌드 시 `/home/[유저이름]/robot_ws/install/my_first_ros_rclpy_pkg/lib/my_first_ros_rclpy_pkg`와 같은 지정 폴더에 실행 파일이 생성됨
``` 
[develop]
script-dir=$base/lib/my_first_ros_rclpy_pkg
[install]
install-scripts=$base/lib/my_first_ros_rclpy_pkg
```


## 2.4. 퍼블리셔 노드 작성

``` python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String


class HelloworldPublisher(Node):

    def __init__(self):
        super().__init__('helloworld_publisher')
        qos_profile = QoSProfile(depth=10) ## depth : 전송 문제 발생 시 퍼블리시할 데이터를 버퍼에 10개까지 저장
        
        ## 퍼블리셔 생성
        ## 토픽 메시지 타입 : String, 토픽 이름 : helloworld, QoS : qos_profile
        self.helloworld_publisher = self.create_publisher(String, 'helloworld', qos_profile)
        
        ## create_timer 함수를 이용한 콜백 함수(publish_helloworld_msg) 실행
        ## timer_period_sec(1)의 시간마다 지정한 콜백 함수를 싱행
        self.timer = self.create_timer(1, self.publish_helloworld_msg)
        self.count = 0

    def publish_helloworld_msg(self):
        msg = String()
        msg.data = 'Hello World: {0}'.format(self.count) ## msg.data에 실제 데이터 저장
        self.helloworld_publisher.publish(msg) ## 토픽 퍼블리시
        self.get_logger().info('Published message: {0}'.format(msg.data)) ## 터미널 창에 출력해주는 함수
        self.count += 1


def main(args=None):
    rclpy.init(args=args) ## 초기화
    node = HelloworldPublisher()
    try:
        rclpy.spin(node) ## 생성한 노드를 spin시켜 지정된 콜백 함수가 실행될 수 있도록 함
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 2.5. 서브스크라이버 노드 작성
``` python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String


class HelloworldSubscriber(Node):

    def __init__(self):
        super().__init__('Helloworld_subscriber')
        qos_profile = QoSProfile(depth=10)
        
        ## 서브스크라이버 생성
        ## 토픽 메시지 타입 : String, 토픽 이름 : helloworld, 콜백 함수 : subscribe_topic_message, QoS : qos_profile        
        self.helloworld_subscriber = self.create_subscription(
            String,
            'helloworld',
            self.subscribe_topic_message,
            qos_profile)

    ## 서브스크라이브한 메시지를 출력해주는 콜백 함수
    def subscribe_topic_message(self, msg):
        self.get_logger().info('Received message: {0}'.format(msg.data))


def main(args=None):
    rclpy.init(args=args)
    node = HelloworldSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 2.6. 빌드
- colcon 빌드 툴을 사용한 ROS2 패키지 빌드
  ``` bash
  $ cd ~/ROS2_ws
  $ colcon build --symlink-install ## 전체 패키지 모두 빌드
  $ colcon build --symlink-install --packages-select <Package Name> ## 특정 패키지만 빌드
  $ colcon build --symlink-install --packages-up-to <Package Name> ## 특정 패키지 및 해당 패키지의 의존성 패키지까지 함께 빌드

  ## Example
  $ colcon build --symlink-install --packages-select my_first_ros_rclpy_pkg
  ```
- 특정 패키지 첫 빌드 후 환경설정 파일을 불러와 실행 가능한 패키지의 노드 설정을 해줘야 빌드된 노드 실행 가능
  ``` bash
  $ . ~/ROS2_ws/install/local_setup.bash
  ```

#### 2022/04/09
# 3. ROS 프로그래밍 기초(C++)
## 3.2. 패키지 생성
```
$ cd ~/robot_ws/src/
$ ros2 pkg create my_first_ros_rclcpp_pkg --build-type ament_cmake --dependencies rclcpp std_msgs
```

## 3.3. 패키지 설정
### 3.3.1. 패키지 설정 파일(package.xml)
- C++ => build_type: ament_cmake
``` xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_first_ros_rclcpp_pkg</name>
  <version>0.0.1</version>
  <description>ROS 2 rclcpp basic package for the ROS 2 seminar</description>
  <maintainer email="pyo@robotis.com">Pyo</maintainer>
  <license>Apache License 2.0</license>
  <author>Mikael Arguedas</author>
  <author>Morgan Quigley</author>
  <author email="jacob@openrobotics.org">Jacob Perron</author>
  <author email="pyo@robotis.com">Pyo</author>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### 3.3.2. 빌드 설정 파일(CMakeLists.txt)
``` cmake
# Set minimum required version of cmake, project name and compile options
cmake_minimum_required(VERSION 3.5)
project(my_first_ros_rclcpp_pkg)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Build
add_executable(helloworld_publisher src/helloworld_publisher.cpp)
ament_target_dependencies(helloworld_publisher rclcpp std_msgs)

add_executable(helloworld_subscriber src/helloworld_subscriber.cpp)
ament_target_dependencies(helloworld_subscriber rclcpp std_msgs)

# Install
install(TARGETS
  helloworld_publisher
  helloworld_subscriber
  DESTINATION lib/${PROJECT_NAME})

# Test
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Macro for ament package
ament_package()
```

## 3.4. 퍼블리셔 노드 작성
``` cpp
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;


class HelloworldPublisher : public rclcpp::Node
{
public:
  HelloworldPublisher()
  : Node("helloworld_publisher"), count_(0)
  {
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
    helloworld_publisher_ = this->create_publisher<std_msgs::msg::String>(
      "helloworld", qos_profile);
    timer_ = this->create_wall_timer(
      1s, std::bind(&HelloworldPublisher::publish_helloworld_msg, this));
  }

private:
  void publish_helloworld_msg()
  {
    auto msg = std_msgs::msg::String();
    msg.data = "Hello World: " + std::to_string(count_++);
    RCLCPP_INFO(this->get_logger(), "Published message: '%s'", msg.data.c_str());
    helloworld_publisher_->publish(msg);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr helloworld_publisher_;
  size_t count_;
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HelloworldPublisher>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
```

## 3.5. 서브스크라이버 노드 작성
``` cpp
#include <functional>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using std::placeholders::_1;


class HelloworldSubscriber : public rclcpp::Node
{
public:
  HelloworldSubscriber()
  : Node("Helloworld_subscriber")
  {
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
    helloworld_subscriber_ = this->create_subscription<std_msgs::msg::String>(
      "helloworld",
      qos_profile,
      std::bind(&HelloworldSubscriber::subscribe_topic_message, this, _1));
  }

private:
  void subscribe_topic_message(const std_msgs::msg::String::SharedPtr msg) const
  {
    RCLCPP_INFO(this->get_logger(), "Received message: '%s'", msg->data.c_str());
  }
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr helloworld_subscriber_;
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HelloworldSubscriber>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
```

## 3.6. 빌드
- my_first_ros_rclcpp_pkg 패키지만 빌드
  ``` bash
  $ cd ~/robot_ws
  $ colcon build --symlink-install --packages-select my_first_ros_rclcpp_pkg
  ```
- 빌드 시 `ModuleNotFoundError: No module named 'catkin_pkg'` 에러가 발생하였는데, 이 때 필요한 패키지 설치 후 제대로 빌드되는 것을 확인하였다.
  ``` bash
  pip install catkin_pkg
  ```
- 특정 패키지의 첫 빌드 후에는 아래와 같이 환경설정 파일을 불러와서 실행 가능한 패키지의 노드 설정을 해야 빌드된 노드 실행 가능
  ``` bash
  $ source ~/robot_ws/install/local_setup.bash
  또는
  $ . ~/robot_ws/install/local_setup.bash
  ```