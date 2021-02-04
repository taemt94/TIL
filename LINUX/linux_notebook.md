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