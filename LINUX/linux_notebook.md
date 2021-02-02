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