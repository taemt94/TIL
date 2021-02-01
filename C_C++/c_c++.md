# 0103
### C/C++

``` c
kmap::KMap * map;
map = new kmap::KMap(data, nat, id);
map->init();
```
- namespace 및 동적 할당 예시
	- kmap이라는 namespace에 작성된 KMap 클래스 객체 포인터 map 선언 후 동적 할당.
	- 동적 할당하면서 매개변수를 가진 클래스 생성자 호출.
	- map은 객체 포인터이므로 멤버 함수 호출 시 '->' 사용.  
  

``` c
#include <vector>

std::vector<pth*> startPth;
int id = startPth[0]->getId();
```
- vector 컨테이너 활용 예시
	- vector 객체를 생성할 때는 <> 내에 다루고자 하는 타입을 지정한다.
	- pth라는 클래스의 포인터로 타입을 지정하였으므로, vector 원소의 포인터 접근 방식으로 클래스의 멤버에 접근 가능하다.

``` c
#include <stdio.h>
#include <stdint.h>

int main(){
	int8_t num1 = -128;
	uint32_t num2 = 4294967295;
	uint32_t num3 = UINT32_MAX;
}
```
- uint32_t 예시
	- 기존의 C언어에서 정수형은 short, int, long으로 나누어지지만, 이러한 자료형 이름이 많은 혼란을 가져와 C99 표준부터 stdint.h 헤더 파일이 추가되었다.
	- 정수 자료형에는 int8_t, int16_t, int32_t, int64_t가 있고, 자료형 이름에 비트 단위로 크기가 표시된다.
	- 부호가 없는 자료형은 앞에 'u'를 붙이면 된다.
	- 이렇게 하면 자료형의 크기를 명시적으로 알 수 있기 때문에 유용하므로 프로그래밍 시 stdint.h를 사용하는 것이 좋다.
	- stdint의 최소, 최대값은 예시와 같이 사용할 수 있고, 부호가 있는 경우 'U'를 빼면 된다.

---
# 0104
### C/C++
```c
#include <time.h>

clock_t start, end;
start = clock();
/*
시간 측정할 코드
*/
end = clock();
double result = ((double)end - start);

struct timespec th_start, th_end;
clock_gettime(CLOCK_MONOTONIC, &th_start);
/*
시간 측정할 코드
*/
clock_gettime(CLOCK_MONOTONIC, &th_end);
double th_result = (th_end.tv_sec - th_start.tv_sec) + (th_end.tv_nsec - th_start.tv_nsec) / 1000000000.0);
```
- 일반적으로 windows의 C++ 환경에서 시간을 측정할 때에는 clock() 함수를 사용하면 된다.
- 그러나 Linux 환경에서 pthread를 사용하여 다중 프로세싱을 할 경우 각각의 쓰레드의 시간을 clock()으로 측정하면 부정확한 값이 측정된다.
- clock()의 경우 하나의 쓰레드의 시간을 측정할 때 다른 쓰레드도 동작하고 있으면 다른 쓰레드에 대한 클럭도 함께 올리기 때문이라고 한다.
- 따라서 다중 프로세싱을 사용할 경우, clock() 대신 위와 같이 clock_gettime()를 사용한다.
---

# 0105
### C/C++
``` c
double a_total[5] = {0, };
Circle cArray[3] = {Circle(10), Circle(), Circle(5)};
```
- 배열을 선언할 때 값을 같은 값으로 초기화하고 싶을 경우 for문을 사용할 필요 없이 위와 같이 중괄호 안에 초기값을 쓰면 된다.
- 객체 배열의 경우에는 생성자를 위와 같이 중괄호 안에 써주면 된다.

# 0108

### C/C++
``` c
	std::cout << "double 타입의 최대 값:  " << std::numeric_limits<double>::max( ) << std::endl;
	std::cout << "double 타입의 최소 값:  " << std::numeric_limits<double>::min( ) << std::endl;
```
- 나머지 데이터형은 <double> 부분에서 데이터형만 바꿔주면 된다.

# 0113
### C/C++
``` c
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> path;
    for (int i = 0; i < minipath.size(); i++)
        path.pushback(minipath.getlinkId()); 
    
    std::reverse(path.begin(), path.end());

    for (int i = 0; i < path.size(); i++){
        printf("[%d]\n", path[i]);
    }
}
```
- 오랜만에 vector 컨테이너와 algorithm의 reverse 함수를 사용해서 간단하게 정리하였다.

# 0120
### C/C++
- Visual Studio 2019 디버거 사용법
    1. Break point를 찍고서 F5(디버깅 시작)을 누르면 break point까지만 실행된다.
    2. Break point에 도착했을 때 F11(한 단계씩 코드 실행)을 누르면 한 줄씩 코드가 진행된다.
    3. Break point를 우클릭한 후 '조건'을 클릭하면 조건식을 추가할 수 있는데, 조건식을 추가하면 조건식을 만족할 때에만 break point에서 멈춘다.
    4. F10(프로시저 단위 실행)을 누르면, 함수 디버깅에서 함수 내부로 들어가지 않고 함수 자체를 실행한다.
    5. F11을 눌렀을 때 함수 내부로 들어가게 되었을 경우, shift + F11을 누르면 함수를 빠져나온다.
    6. 조사식에 추적을 원하는 변수를 입력하면 추적해야 되는 변수의 값을 확인할 수 있다.
    7. 변수의 주소값을 확인하고 싶을 때는 주소값 입력할 때와 마찬가지로 변수 앞에 &를 붙이면 된다.

# 0121
### C/C++
``` c
#include <iostream>
#include <chrono>

std::chrono::system_clock::time_point st, end;
std::chrono::duration<double> timer;

st = std::chrono::system_clock::now();
vecAdd <<<1, NUM_DATA>>>(a, b, c);
end = std::chrono::system_clock::now();
timer = end - st;
printf("%f\n", timer);
```
- chrono를 사용하면 나노세컨드 단위로 시간 측정이 가능하다.

# 0128
### C/C++
`gcc main.c`  
`g++ .\main.cpp`  
- vscode를 사용할 때 터미널에 위와 같이 입력하면 컴파일러가 소스코드를 읽어서 실행코드로 바꿔준다.

`gcc main.c -o main.exe`  
`g++ .\main.cpp -o main.exe`  
- 실행 파일 이름을 -o 옵션으로 원하는 대로 설정할 수 있다.

`gcc -c main.c`  
`g++ -c .\main.cpp`  
- -c 옵션을 주면 컴파일만 하겠다는 뜻으로, 컴파일 후 object 파일을 생성해준다.

`gcc main.o -o exe_from_obj.exe`  
`g++ .\main.o -o exe_from_obj.exe`  
- object 파일만 생성하면 실행 파일은 생성이 안되므로, 컴파일러를 통해 object 파일로 실행 파일을 생성한다.