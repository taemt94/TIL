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
