# 01/04
### LINUX
`unzip -l compressed.zip`
- 압축을 해제하지 않고 압축 파일 내의 목록만 출력한다.  

`unzip compressed.zip -d /path/to/put`

- 압축 파일을 압축 해제한다.
- -d 옵션을 사용하면 원하는 디렉토리에 압축 해제할 수 있다.
---
# 01/06
### LINUX
`zip -r src.zip src`
- 압축파일을 생성할 때 -r 옵션을 주면 src 폴더와 src 폴더의 모든 하위 폴더를 모두 압축한다.
- -r 옵션을 안주면 하위 폴더를 제외한 src 폴더에 있는 파일만 압축한다.  

`top`
- 서버 컴퓨터 user 정보와 각 user가 CPU를 몇 % 점유하는지 확인할 수 있다.
- 이것 외에도 VIRT, RES 등 다양한 정보가 출력되는데, 아직 잘 모르는 부분이므로 차차 알아가기로 한다.  

# 01/16
### LINUX
- mv [이동할 파일명][이동할 위치]
    : 파일의 위치를 변경할 때 사용하는 명령어.  
    : 예) mv /home/index.html /home/test/index2.html  
        -> index.html이라는 파일을 test 폴더로 옮기면서 index2로 파일명도 바꿀 수 있다.  
- cp -[option][복사할 파일명][복사할 위치]
    : 파일을 복사하여 새로운 파일을 만들 때 사용하는 명령어.  
    : 예) cp /home/index.html /home/test/index2.html  
    -> index.html이라는 파일을 test 폴더로 옮기면서 index2로 파일명도 바꿀 수 있다.  

# 0119
### LINUX
`sudo sed -i 's/BIG-REQUESTS/_IG-REQUESTS/' /usr/lib/x86_64-linux-gnu/libxcb.so.1`  
- 리눅스에서 vscode 설치 후 터미널 창에서 code 명령어를 입력하였는데도 창이 뜨지 않을 경우 위의 명령어를 입력한다.  

`mv /home/test/ /home/test2`  
- mv 명령어의 입력이 둘 다 디렉토리인 경우 디렉토리의 이름이 변경된다.
- 파일명도 마찬가지의 방법으로 이름을 변경할 수 있다.

`grep 'cpu cores' /proc/cpuinfo | tail -1`  
- CPU의 물리 코어수를 확인하는 명령어.  

`grep processor /proc/cpuinfo | wc -l`  
- CPU의 가상 코어수를 확인하는 명령어.
- 가상 코어는 하이퍼 스레딩 테크놀로지 HTT 기술로, 물리적인 코어에 가상의 스레드를 추가하여 다중 작업에 유리하도록 만드는 기술이라고 한다.
- 듀얼코어일 때 가상의 2 스레드를 사용하면 트리플 코어 정도의 성능을 낸다고 한다.
- 프로젝트에서 CPU 코어를 사용하여 멀티스레딩을 할 일이 있었는데, 서버 컴퓨터의 CPU 물리 코어 수가 18개, 가상 코어 수가 36개라서 코어를 18개까지 사용할 수 있다는 건지 36개까지 사용할 수 있다는 건지 테스트를 해보았다.
- 테스트 결과 36개의 코어를 모두 사용할 수 있는 것 같다.

# 0121

### LINUX
`fusermount -u ~/gdrive`  
- 리눅스에서 구글 드라이브 언마운트할 때 사용하는 명령어.  

# 02/02
`ps`  
- 현재 실행되고 있는 프로세스 목록 확인 
 
`ps -f`  
- 프로세스의 자세한 정보 확인  

`ps -ef`  
- 모든 프로세스 리스트 확인  
---
### VNC VIEWER
- vnc viewer를 사용할 때 windows - linux 간 복사/붙여넣기가 되지 않을 경우 `autocutsel`을 아래와 같이 설치하면 된다고 한다.
    1. `sudo apt-get install autocutsel`
    2. `autocutsel -s PRIMARY -fork`
- `autocutsel`을 설치하고 나면, 터미널 창에서 `SHIFT + INSERT`로 붙여넣기를 하면 된다.
---
`apt-get install`  
- 서버 컴퓨터에서 위의 명령어로 설치하는 프로그램은 서버 전체에 적용되는 것이라고 한다.
- 그래서 CUDA 드라이버나 nvcc(cuda-toolkit)은 충돌 가능성이 있기 때문에 함부로 내 계정에서 다시 깔지 않는 것이 좋다.

# 2021/03/02
### tee 명령어
- `tee` 명령어는 실행 파일의 출력 결과를 화면에 나타냄과 동시에 파일에 입력하도록 해주는 명령어이다.  
`./route.sh | tee  [-a][-i][Filename.txt]`  
- 사용법은 위와 같이 실행 파일 뒤에 tee 명령어를 입력한 후 옵션과 출력 결과를 입력할 파일명을 써주면 된다.
- `-a` 옵션은 파일에 덮어쓰기를 하지않고 해당 파일에 추가하여 입력한다.
- `-i` 옵션은 interrupt를 무시하는 옵션이다.

# 2021/03/09
### 권한 설정
- 프로그래밍을 하면서 파일 입출력을 하는 과정에 Segmentation fault(C 기준)가 발생하며 코드 실행이 안될 때가 있다.
- 이 때에는 생성하거나 불러오려는 파일이나 디렉토리의 권한을 확인하자.
- 서버에서 파일을 수없이 주고 받는 과정에서 권한이 허용되어 있지 않은 경우가 존재하므로 이때에는 모든 권한을 주도록 해주기만 하면 문제가 쉽게 해결된다.
    >`chmod 777 [directory]`

# 2021/03/17
### 권한 설정
- `chmod`를 사용하여 모든 하위 폴더와 하위 파일에 권한 설정을 하고 싶을 때에는 `-R` 옵션을 적용하면 된다.
    >`chmod -R [8bit permission][directory]`

# 2021/03/23
### 경로 확인
1. `which`
   - 특정 명령어의 위치를 찾아주는 명령어이다.
     > ~$ `which find`  
     > /bin/find
   - `-a` 옵션을 주면 검색 가능한 모든 경로에서 해당 명령어를 찾는다.
     > ~$ `which -a find`  
     > /bin/find  
     > /usr/bin/find
2. `whereis`
   - 명령어의 실행 파일 위치, 소스 위치, man 페이지 파일의 위치를 찾아주는 명령어이다.
   - man 페이지 파일은 아직 뭔지 잘 모르겠지만, 추후에 사용하게 되면 좀 더 자세히 찾아보기로 한다.
     > ~$ `whereis nvidia-smi`  
     > nvidia-smi: /usr/bin/nvidia-smi /usr/share/man/man1/nvidia-smi.1.gz
3. `locate`
   - 다양한 패턴의 파일들을 찾을 때 유용하게 사용될 수 있는 명령어이다.
   - 아래와 같이 디렉토리명을 입력하면 해당 디렉토리 내의 모든 디렉토리를 확인할 수 있다.
      > ~$ `locate ~/project/test`  
      > /home/taesankim/project/test  
      > /home/taesankim/project/test/a.out  
      > /home/taesankim/project/test/test  
      > /home/taesankim/project/test/test.cpp  
    - 또는 디렉토리 내의 특정 확장자를 가지는 모든 파일을 찾을 수도 있다.
      > ~$ `locate ~/project/*.bin` 
    - `-n 10` 과 같은 옵션을 주면 지정한 개수만큼 검색되도록 할 수도 있다.
      > ~$ `locate -n 10 ~/project/*.bin` 

# 2021/03/24
### 네트워크 확인 시 유용한 명령어들
`ifconfig`
  - "interface configuration"의 약자로 리눅스 네트워크 관리를 위한 인터페이스 구성 유틸리티이다.
  - 현재 네트워크 구성 정보를 표시해준다.
  - 네트워크 인터페이스에 IP 주소, 넷 마스크 또는 broadcast 주소를 설정하고, 인터페이스의 별칭을 만들거나 하드웨어 주소 설정, 인터페이스 활성화 및 비활성화 등을 할 수 있다고 한다.  
  
`ifconfig -a`  
  - `-a` 옵션을 주면 비활성화된 네트워크 인터페이스도 모두 볼 수 있다.  

`ifconfig [interface name] down`  
  - [interface name]에 해당하는 인터페이스를 비활성화한다.
  - 연결을 해제하는 것으로 생각하면 될 것 같다.

`ifconfig [interface name] up`  
  - [interface name]에 해당하는 인터페이스를 활성화한다.

`sudo ethtool -p [interface(port) name]`  
  - PC에 이더넷 포트가 여러 개 있을 때 어느 것이 어느 이름을 가지고 있는지 알 수 없을 때 사용하면 좋은 방법이다.
  - 인터페이스 이름을 입력하여 명령어를 실행하면 해당 이너넷 포트의 LED가 깜빡깜빡 거린다.

# 2021/03/30
### Ubuntu 버전별 이름
- Ubuntu 18.04는 Ubuntu Bionic, Ubuntu 20.04는 Ubuntu Focal로도 불린다.
- ROS를 리눅스 노트북에 새로 설치하는 과정에서 Bionic과 Focal이라는 이름을 처음 알게되었다.
- 자잘자잘한 것이라도 배운거니까 일단 적어둔다.

### IP 간단 확인
```
$ hostname -I
```

# 2021/04/05
### Ubuntu 재설치
- 차량용 컴퓨터(nuvo-6108gc)에 깔려 있는 리눅스가 16.04 버전이기도 하고, 워낙 컴퓨터를 함부로 쓰다보니 정리가 전혀 되어있지 않아 Ubuntu 18.04로 재설치하기로 하였다.
- 아래의 링크를 참조하여 설치하였다.  
[Ubuntu installation](https://medium.com/code-states/%EB%AC%B4%EC%9E%91%EC%A0%95-%EC%9A%B0%EB%B6%84%ED%88%AC-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0-65dae2631ecc)  
[Ubuntu installation2](https://keepdev.tistory.com/69)  
- 참고로, 차량용 컴퓨터는 부팅 시 F2를 눌러 바이오스에 진입하였을 때는 부팅 순서를 변경하는 옵션이 없었다.
- 부팅 순서 변경을 할 때에는 F8(F8~F12 중 하나, 아마 F8이었던 것 같다.)을 눌러주면 순서를 변경할 수 있는 창이 나온다.

### 설치 후 기본 세팅
- build-essential 설치
  - build-essential을 설치하면 기본적으로 make, gcc, g++ 등이 포함되어 설치된다.
  ```
  $ sudo apt-get update
  $ sudo apt-get install build-essential
  $ sudo apt-get install git
  $ git config --global user.gmail "you@example.com"
  ```
  [reference](https://conservative-vector.tistory.com/entry/Ubuntu-Install)

- nvidia driver 설치
  - `ubuntu-drivers` 명령어를 통해 추천 드라이버 확인
  ```
  $ ubuntu-drivers devices
  == /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
  modalias : pci:v000010DEd00001E87sv00003842sd00002182bc03sc00i00
  vendor   : NVIDIA Corporation
  driver   : nvidia-driver-418-server - distro non-free
  driver   : nvidia-driver-450 - distro non-free
  driver   : nvidia-driver-460-server - distro non-free recommended
  driver   : nvidia-driver-460 - distro non-free
  driver   : nvidia-driver-450-server - distro non-free
  driver   : xserver-xorg-video-nouveau - distro free builtin
  ```
  - 필요한 repository 추가
  ```
  $ sudo add-apt-repository ppa:graphics-drivers/ppa
  $ sudo apt update
  ```
  - 설치 가능한 드라이버 목록 출력
  ```
  $ apt-cache search nvidia | grep nvidia-driver-460
  ```
  - apt로 nvidia 드라이버 설치
  ```
  $ sudo apt-get install nvidia-driver-460
  ```
  [reference](https://codechacha.com/ko/install-nvidia-driver-ubuntu/)

- 그 외 다른 패키지 설치
  ```
  $ sudo apt-get install python-pip  ## Python 2.x pip
  $ sudo apt-get install python3-pip ## Python 3.x pip

  $ sudo apt-get install net-tools   ## ifconfig 사용
  
  $ sudo apt-get install can-utils   ## Linux CAN(aka SocketCAN) subsystem userspace utilities and tools
  ```

### 리눅스 모듈 관리: modprobe
- 리눅스 개발자들은 가급적 사용자가 시스템과 관련된 내용을 잘 몰라도 사용할 수 있게끔 많은 기능을 리눅스 커널에 포함시켰는데, 그 중 하나가 커널에서 하드웨어를 발견했을 때 필요한 모듈을 자동 적재하는 기능이다.
- 모듈 자동 적재는 모듈 유틸리티인 modprobe 프로그램을 통해 리눅스 커널이 적재 요청한 모듈이 동작할 수 있도록 이에 필요한 부수적인 모듈을 커널에 차례로 먼저 등록해준다.
- `modprobe` 명령어는 모듈의 의존성을 고려하여(설치한 모듈을 사용하기 위해 필요한 모듈들도 함께 적재) 리눅스 커널에 모듈을 적재하는 명령어이다.
  ```
  $ sudo modprobe can
  $ sudo modprobe kvaser_usb
  ```
- 위와 같이 입력하면 각각의 모듈을 리눅스 커널에 적재한다.
  - 주요 옵션:
    ```
    -l: 사용 가능한 모듈 정보 출력
    -r: 의존성을 고려하여 모듈을 제거
    -c: 모듈 관련 환경 설정 파일 내용 출력 
- 모듈 정보 확인
  ```
  $ sudo modinfo [모듈명]
  ```

# 2021/08/25
### TouchScreen Calibration on Ubuntu 20.04
- 차량용 컴퓨터의 모니터로 사용하던 터치 모니터가 Ubuntu 20.04로 버전업을 하고 나서 터치 입력이 제대로 되지 않기 시작하였다.
- 처음에는 모니터 문제인줄 알고 Windows 컴퓨터로 가져와서 테스트를 해봤는데, Windows 컴퓨터에서는 터치가 제대로 동작하였다.
- 구글링 해본 결과, 차량에서 터치 모니터를 사용하기 위해 화면 방향을 세로 방향(Portrait Right)으로 변경하였는데, 이 때 터치 입력의 좌표값이 제대로 calibrate되지 않아서 생기는 문제였다.
- 해당 문제는 아래와 같이 해결하면 된다.
  ```
  $ xinput list ## Input device들의 목록을 보여준다.
  ```
- 위의 명령어를 통해 나온 리스트로 터치 모니터의 이름을 확인해야 한다.
- 터치 모니터의 경우 누구나 식별하기 쉬운 이름을 가지고 있다.

  ```
  $ xinput list-props "Touch Monitor name.." | grep "Coordinate Transformation Matrix"
  ```
- 위의 명령어로 현재 터치 모니터 터치 입력의 좌표 변환 행렬값(Coordinate Transformation Matrix, CTM)을 확인할 수 있다.
- 모니터의 가장 기본 방향인 가로(Landscape) 방향의 경우 CTM은 Identity 행렬이다.
- 확인 결과, 모니터의 방향은 세로 방향으로 변경되었더라도 이 CTM이 기본 identity 행렬이라서 터치 입력의 좌표가 변경되지 않았던 것이었다.
- 이러한 경우 Linux 명령어를 통해 터치 입력의 좌표축을 변경해주어야 한다.
- Linux 명령어를 통해 모니터의 방향 및 터치 인풋의 방향을 변경하는 방법은 아래와 같다.

- 모니터 방향을 왼쪽 세로 방향(Portrait Left, Clockwise $$90^\circ$$)으로 변경하는 경우
  ```
  $ xrandr -o left ## 모니터 방향을 Portrait Left로 변경
  $ xinput set-prop "Touch Monitor name.." "Coordinate Transformation Matrix" 0 -1 1 1 0 0 0 0 1 ## 터치 입력의 좌표축 변경
  ```
- 위의 두번째 명령어 마지막의 숫자는 CTM을 나타내는 3*3 크기의 행렬을 의미한다.

- 모니터 방향을 오른쪽 세로 방향(Portrait Right, Counterclockwise $$90^\circ$$)으로 변경하는 경우
  ```
  $ xrandr -o right ## 모니터 방향을 Portrait Right로 변경
  $ xinput set-prop "Touch Monitor name.." "Coordinate Transformation Matrix" 0 1 0 -1 0 1 0 0 1 ## 터치 입력의 좌표축 변경
  ```

- 모니터 방향을 가로 반대 방향(Landscape Invert, Clockwise $$180^\circ$$)으로 변경하는 경우
  ```
  $ xrandr -o inverted ## 모니터 방향을 Landscape Invert로 변경
  $ xinput set-prop "Touch Monitor name.." "Coordinate Transformation Matrix" -1 0 1 0 -1 1 0 0 1 ## 터치 입력의 좌표축 변경
  ```

- 추가적으로, `xinput` 명령어를 사용하여 좌표축을 변경할 때 같은 터치 모니터가 여러 개라는 경고가 뜨면서 좌표축 변경이 되지 않는 경우가 있다.
- 실제로 `xinput list`를 통해서 확인해보아도 id만 다른 같은 이름의 터치 모니터가 2개인 것을 보았다.
- 이러한 경우 `xinput set-prop` 명령어를 사용할 때 터치 모니터 이름을 입력하는 위치에 이름 string 대신 id(int)를 입력해주면 된다.
- 해당 모니터의 경우 같은 모니터 이름에 2개의 id가 나와서 두 id에 대해 모두 `xinput set-prop` 명령어를 사용하였고 그 결과 터치 입력이 제대로 동작하였다.

- Reference : <https://wiki.ubuntu.com/X/InputCoordinateTransformation>

# 2021/09/01
### Sound output device default setting on Ubuntu
- 우분투 상에서 PC에 연결된 스피커가 여러개일 경우 기본 스피커 설정하는 방법
  - 아래의 명령어를 터미널 상에 치면 현재 연결되어 있는 device들을 확인할 수 있다.
  - 여기서 기본값으로 사용할 스피커의 이름을 복사한다.
  ```
  pactl list short sinks
  ```
  - 아래의 명령어를 사용하면 원하는 기기로 기본 스피커를 설정할 수 있다.
  ```
  pactl set-default-sink <Device_Name>
  ```
  - 또한, Ubuntu의 Startup Applications을 사용하면 PC의 전원이 켜질 때마다 위의 명령어를 통해 기본 스피커가 설정되도록 할 수 있다.
  - Ubuntu에서 Startup Applications을 검색하여 열고, "Add"를 누른 후 'command' 위치에 위의 명령어를 입력하고 "Save"를 눌러 startup application을 추가해주면 된다.

# 2021/09/30
### Install .deb/.rpm file on Ubuntu
- 확장자가 deb이거나 또는 rpm인 파일은 리눅스에서 사용하는 프로그램 설치 패키지이다.
- 이러한 파일들을 사용하여 프로그램을 설치하는 방법은 아래와 같다.
1. deb 파일 설치
  - deb 파일은 데비안 꾸러미 파일이고, 이는 `dpkg` 명령어를 사용하여 설치한다. 
  ```
  $ dpkg -i [file_name.deb]
  ```
  - 설치한 프로그램을 제거할 때에는 `-r` 옵션을 사용하면 된다.
  ```
  $ dpkg -r [file_name.deb]
  ```
2. rpm 파일 설치
  - rpm 파일은 레드햇 패키지 관리자에서 사용되는 파일이다.
  - 우분투 상에서는 rpm 파일을 이용하여 프로그램을 설치하는 것을 권장하지 않는다고 한다. 
  - 보통 rpm 파일이 존재할 경우 deb 파일도 존재하기 때문에 크게 문제가 되지 않지만, 혹시 rpm 파일을 꼭 사용해야 할 때에는 `alien` 명령어를 사용하여 rpm 파일을 deb 파일로 변환하면 된다.
  ```
  $ alien [file_name.rpm]
  ```