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


# 02/08
``` c
#include <stdio.h>

int main() {
	int a, b;
	a = 123;

	int *a_ptr;
	a_ptr = &a;
	printf("%d %d %p\n", a, *a_ptr, a_ptr);

	b = a_ptr;
}
```
- 포인터의 주소값을 출력할 때는 `%p` 형식 지정자를 사용한다.
- `b = a_ptr`의 경우 권장하는 방법은 아니지만, c의 철학에 따라 컴파일러가 에러 대신 경고를 준다고 한다.
- 프로그래머가 (int) 형으로 형변환하려는 의도일수도 있기 때문이다.
- 그러나 절대 권장하는 방법은 아니고, 고급 프로그래머도 이렇게 사용하는 경우는 드물다고 한다.

``` c
#include <stdio.h>
int main(){
	int *a, b;
	int* a, b;
	int *a; 
	int *b;
}
```
- 첫번째 줄의 경우 a는 int형 포인터이고, b는 int형 변수이다.
- 두번째 줄에서 `*`을 int 바로 옆에 붙인다고 해서 b까지 포인터가 되는 건 아니고, 첫번째 줄과 동일한 의미이다.
- 권장하는 방법은 세번째, 네번째 줄과 같이 줄바꿈을 통해 나눠 사용하는 것이다.

``` c
#include <stdio.h>
int main(){
	int *ptr = 1234;
	printf("%p\n", ptr);
	printf("%d\n", *ptr);

	int *safer_ptr = NULL; // nullptr
	int a = 1;
	int b;
	scanf("%d", &b);
	
	if (b % 2 == 0)
		safer_ptr = &a;

	if(safer_ptr != NULL)
		printf("%p\n", safer_ptr);
		printf("%d\n", safer_ptr);
}
```
- 포인터 변수에 값을 대입하여 초기화하게 될 경우, 컴퓨터가 리디렉션을 하는 과정에서 해당 주소에 사용할 수 있는 값이 없기 때문에 런타임 에러가 발생한다.
- 이러한 경우를 방지하기 위해 포인터를 선언함과 동시에 NULL 포인터를 관습적으로 대입한다.
- C++ 에서는 NULL을 대신하여 nullptr가 추가되어 있다.
- NULL 포인터를 사용함으로써 포인터 변수가 접근할 수 있는 메모리 공간이 지정되어 있지 않아 런타임 에러가 발생하는 일을 위와 같이 방지할 수 있다.
  
``` c
include <stdio.h>

void swap1(int a, int b){
	printf("%p %p\n", &a, &b);

	int temp = a;
	a = b;
	b = temp;
}

void swap2(int *a, int *b){
	printf("%p %p\n", a, b);

	int temp = *a;
	*a = *b;
	*b = temp;
}

int main(){
	int a = 123;
	int b = 456;
	
	printf("%p %p\n", &a, &b);

	swap1(a, b);
	printf("%d %d\n", a, b);

	swap2(&a, &b);
	printf("%d %d\n", a, b);

	return 0;
}
```
- 위에서 main 함수에 있는 a, b와 swap1 함수에 있는 a, b는 전혀 다른 변수이기 때문에 주소를 출력할 경우 전혀 다른 주소가 출력된다.
- 따라서 swap1 함수로 swap을 해봤자 main 함수의 a, b의 값이 swap되지 않고 그대로 인 것을 확인할 수 있다.
- 이를 `값에 의한 호출`이라고 한다.
- swap 함수를 통해 main 함수의 a, b의 값을 바꾸고 싶을 때에는 포인터를 사용하여 swap2 함수와 같이 a, b의 값을 넘겨주는 것이 아니라 a와 b의 주소를 넘겨주면 된다.
- swap2 함수의 경우 a, b의 주소가 main 함수의 a, b의 주소가 동일하다는 것을 알 수 있다.
- 이를 `주소에 의한 호출`이라고 한다.

# 2021/02/18
### 정수와 부동 소수점 숫자의 나눗셈
- 나눗셈 연산자 `/`는 정수와 부동 소수점 숫자에 대해 다른 나눗셈 연산을 수행한다.
- 두 피연산자가 모두 정수일 경우 나눗셈 연산자는 결과값의 정수 부분만 리턴한다.  
ex: `2000 / 1024 -> 1`  
- 두 피연산자 중 하나라도 부동 소수점 숫자인 경우 결과값의 소수 부분까지 리턴한다.  
ex: `2000 / 1024.0 -> 1.953125`, `2000.0 / 1024 -> 1.953125`, `2000.0 / 1024.0 -> 1.953125`

# 2021/02/19
### C++ BEEP sound 
- `printf()`를 호출할 때 `\a` 옵션을 주면 된다.
- putty를 통해 리눅스 상에서도 작동하는 것을 확인하였다.
- 프로젝트를 진행하면서 테스트 단계에 돌입하면 100번 이상 코드를 반복하여 돌려야 하는데, 이를 중간 중간 종료되었는지 확인하는게 매우 귀찮은 작업이라 프로세스가 종료되는 시점에 스피커를 통해 소리를 발생시켜 편의성을 높일 수 있었다.

# 2021/02/28
### 파일 입출력 fopen, fclose
``` c
#include <stdio.h> // C
#include <ctdio>   // C++

int a, b;
int c, d;
FILE *f_ptr = fopen("input.txt", "r");
fscanf(f_ptr, "%d%d", a, b);
fscanf(f_ptr, "%d%d", c, d);
```
- `fopen`을 사용하여 위와 같이 파일의 값을 읽어와 변수에 입력할 수 있다.
- `fopen`에 첫번째 인자는 파일의 디렉토리를 입력하고, 두번째 인자로 "r" 옵션을 주면 파일의 값을 읽어온다.
- FILE 형의 포인터를 선언한 후 `fscanf`를 사용하면 위와 같이 파일의 값을 차례대로 불러와 변수에 입력한다.

``` c
#include <stdio.h> // C
#include <ctdio>   // C++

int a = 5;
int b = 2;
FILE *r_ptr = fopen("output.txt", "w");
fprintf(r_ptr, "a: %d\n", a);
fprintf(r_ptr, "b: %d\n", b);
```
- `fopen`을 사용하여 파일을 생성할 때에는 "w" 옵션을 주면 된다.
- FILE 형의 포인터를 선언한 후 `fprintf`를 사용하면, 위와 같이 "..." 내에 입력한 내용이 파일에 작성된다.
- 같은 파일을 `fopen`을 통해 반복 작성할 경우 기존에 있던 내용에 새로운 내용이 덮어 씌여진다.

#### 2022/02/24
## 단어 입력 받아 출력하기
``` c
#include <stdio.h>

int main(){
    char data[51];
    scanf("%s", data);  ## data 변수는 배열이므로 & 연산자를 붙이지 않아도 된다.
    printf("%s", data);

    return 0;
}
```

## 공백 포함되어 있는 문장 입력 함수
``` c
#include <stdio.h>

int main(){
    char data[2001];
    fgets(data, 2000, stdin);

    return 0;
}
```

## 문자열의 NULL
``` c
#include <stdio.h>

int main(){
    char word[20];
    scanf("%s", word);
    for(int i=0; word[i]!='\0'; i++){
        printf("\'%c\'\n", word[i]);
    }
}
```
- 문자열을 입력으로 받게 되면 문자열의 마지막임을 나타내기 위해 문자열의 마지막에 NULL을 삽입한다.
- NULL은 `\0`으로 표현할 수 있으므로 위와 같이 for 문을 통해 해당 문자가 문자열의 끝인지를 판단할 수 있다.

## printf로 정수 출력하기
``` c
#include <stdio.h>

int main(){
    int y, m, d;
    scanf("%d.%d.%d", &y, &m, &d);
    printf("%02d-%02d-%04d", d, m, y);

    return 0;
}
```
- 2자리 정수를 4자리 칸에 출력하고, 앞 두자리는 0으로 채우고 싶은 경우 `%04d`와 같은 형식을 사용하면 된다.