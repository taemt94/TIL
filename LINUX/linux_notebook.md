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