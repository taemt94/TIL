# 0118
### Python
- python을 설치할 때 환경변수 경로를 설정하지 않고 설치하면, python이 설치되더라도 cmd 창에서는 python을 실행할 수 없다.
- 이를 다시 환경변수에 설정하는 것이 매우 귀찮은 작업이므로, 꼭 python을 설치할 때에는 환경변수를 잘 체크하여 설치하는 것이 좋을 것 같다.

### ANACONDA(Tensorflow 설치 과정)
1. conda update conda
2. conda create -n tensorflow
    - tensorflow 라는 이름의 가상환경 생성
    - 아나콘다에서 base(root)라는 기본 가상환경에 새로운 모듈과 패키지를 설치해도 되지만, 설치 중 오류가 발생할 경우 돌이킬 수 없는 문제가 발생하여 아나콘다를 재설치해야 할 수도 있다고 한다.
    - 따라서 새로운 모듈을 설치할 때는 항상 새로운 가상 환경을 만들고 설치하는 것이 좋다.
3. conda remove -n tensorflow --a
    - 가상환경 삭제 명령어.
4. conda info --envs
    - 가상환경 목록 조회 명령어.
5. activate tensorflow
    - 해당 가상환경 활성화 명령어.
    - 활성화하면 프롬프트 창에서 (base)라고 되어 있던 것이 가상환경의 이름으로 바뀌는 것을 확인할 수 있다.
6. deactivate tensorflow
    - 해당 가상환경 비활성화 명령어.
7. conda install tensorflow
    - 가상환경 상에서 tensorflow 설치.
    - 윈도우 cmd 창에서 모듈을 설치할 때는 pip를 사용하지만, 아나콘다에서는 conda를 사용하는 것이 훨씬 좋다고 한다.

# 0119
### ANACONDA
`conda install conda`  
- 가상환경을 만든 후 conda를 가상환경에도 설치해준다.
- 이 작업을 하지 않으면 CUDA 설치 시 에러가 난다고 한다.  

`conda create --name tf_gpu tensorflow-gpu`  
- tf_gpu라는 이름의 가상환경에 tensorflow-gpu를 입력하여 tensorflow에 필요한 라이브러리들을 자동으로 설치한다.
- 이 명령어를 사용하면 가상환경에 CUDA 툴킷과 cuDNN이 자동으로 설치된다고 하는데, 이렇게 하고서 conda list를 확인해보니 CUDA 툴킷과 cuDNN은 설치되어 있지 않았다.

### ANACONDA Navigator
- Navigator를 사용하면 윈도우 상에서는 아나콘다 프롬프트를 사용할 필요없이 매우 간단하게 필요한 패키지를 설치할 수 있다는 것을 오늘 알게 되었다.
- 패키지 설치 후 버전이 안맞는 문제가 생길 때에도 제거 후 재설치할 필요없이 바로바로 다운그레이드나 업그레이드가 가능하다.
- 다만 Navigator를 실행할 때 반드시 관리자 권한으로 들어가주어야 한다.

# 0122
### ANACONDA  
`conda update -n base conda`  
- Anaconda update 명령어.

# 0129
### PYTHON
`pip show virtualenv`  
- pip로 설치한 모듈의 세부정보를 확인할 수 있는 명령어.  
- virtualenv는 파이썬 가상 환경을 만드는 프로그램이라고 한다.  

`python -m virtualenv tf2env`  
- 윈도우 환경에서 가상환경을 생성할 때에는 다음과 같이 입력해야 한다.  

`source tf2env/bin/activate`  
`call tf2env/scripts/activate`  
- 리눅스에서 가상환경을 활성화할 때는 source 명령어를 사용하면 되는 듯 하다.
- 그러나 윈도우 상에서 가상환경을 활성화할 때에는 call 명령어를 사용해야 한다.


# 2021/04/08
### vars([object])
- `__dict__` attribute를 포함하고 있는 모듈, 클래스, 객체 등의 `__dict__` attribute를 리턴해주는 함수이다. 

# 2021/04/19
### Terminate threads using a flag recieved from keyboard input
- 차량용 컴퓨터에서 CAN 및 다양한 센서들로부터의 데이터를 여러개의 쓰레드를 생성하여 하나의 프로그램에서 받을 수 있도록 하였다.
- 이 때 데이터 수집을 중단하기 위해서 메인 쓰레드에서 키보드 입력(예: 엔터키)를 기다리다가 키보드 입력이 들어오면, 생성된 쓰레드들을 한번에 종료하기 위한 코드가 필요하였다.
- 구글링 해본 결과 파이썬의 threading 패키지는 쓰레드를 강제 종료시키는 함수는 따로 없기 때문에 아래와 같이 flag 변수를 통해 종료시켜야 한다고 한다.
- 아래의 코드에서 `stop_threads`가 flag 역할을 하는 변수이다.
- 메인 쓰레드는 각각의 쓰레드를 생성하고 난 후, 키보드 입력이 들어올 때까지 대기하다가 올바른 키보드 입력이 들어오면 flag 변수를 True로 변경한다.
- flag 변수가 True가 되면 각각의 쓰레드에서 실행되고 있는 반복문이 종료되어 쓰레드 작업도 종료된다.
``` python
import threading
import time

def test_CAN(id, data_name, stop):
    print(f"Thread[{id}] Data_name[{data_name}]: Activated.")
    while True:
        print(f"Thread[{id}] Data_name[{data_name}]: Recieving data started.")
        if stop():
            print(f"Thread[{id}] Data_name[{data_name}]: Recieving data stopped.")
            break

def test_Audio(id, data_name, stop):
    print(f"Thread[{id}] Data_name[{data_name}]: Activated.")
    while True:
        print(f"Thread[{id}] Data_name[{data_name}]: Recieving data started.")
        if stop():
            print(f"Thread[{id}] Data_name[{data_name}]: Recieving data stopped.")
            break

def test_Video(id, data_name, stop):
    print(f"Thread[{id}] Data_name[{data_name}]: Activated.")
    while True:
        print(f"Thread[{id}] Data_name[{data_name}]: Recieving data started.")
        if stop():
            print(f"Thread[{id}] Data_name[{data_name}]: Recieving data stopped.")
            break

def main():
    stop_threads = False
    workers = []
    data_names = ['CAN', 'Audio', 'Video']
    thread_function = [test_CAN, test_Audio, test_Video]
    for id, (d_name, th_func) in enumerate(zip(data_names, thread_function)):
        worker = threading.Thread(target=th_func, args=(id, d_name, lambda: stop_threads))
        workers.append(worker)
        worker.start()
    
    print("Press 'Enter' if you want to terminate every processes.")
    terminate_signal = input()
    while terminate_signal != '':
        print("Wrong input! Press 'Enter'")
        terminate_signal = input()
    if terminate_signal == '':
        stop_threads = True

    for worker in workers:
        worker.join()
    print("Main thread finished.")

if __name__ == "__main__":
    main()
```

- 파이썬 threading 패키지를 이용하는 방법 외에도 multiprocessing 패키지를 이용하여 process.terminate() 함수를 사용할 수도 있다고 한다.
- 위의 코드가 제대로 실행되지 않을 경우 multiprocessing 패키지를 이용하여 구현해볼 예정이다.