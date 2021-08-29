# 2021/08/26
### PCL vs Open3D vs Matlab
- 3D point cloud 데이터를 처리하기 위한 라이브러리는 대표적으로 PCL, Open3D, Matlab Computer Vision Toolbox 등이 있다.
1. PCL(Point Cloud Library)
   - PCL은 2011년부터 배포되어 있는 라이브러리로, 3D 데이터를 처리하기 위한 다양한 기능들을 제공한다.
   - PCL에서 제공하는 주요 기능은 아래과 같다.
     - 데이터에서 이상값과 노이즈 제거 등의 필터링 
     - 점군 데이터로부터 3D 특징/특징점 추정을 위한 자료 구조와 방법들 
     - 여러 데이터셋을 합쳐 큰 모델로 만드는 정합(registration) 작업 
     - 점군으로부터 클러스터들로 구분하는 알고리즘 
     - 선, 평면, 실린더 등의 모델 계수 추정을 위한 형태 매칭 알고리즘 
     - 3D 표면 복원 기법들
   - 그러나 PCL의 경우 최근에는 업데이트가 제대로 이루어지지 않는다고 한다.
   - 또한, 의존성(Dependency) 문제로 인해 설치하기가 매우 어렵다는 문제가 있다. 최소 대여섯개의 3rd Party 라이브러리를 설치해주어야 하기 때문에 많은 시간이 필요하고, 필자도 설치를 시도하다가 능력의 한계로 인해 결국 설치를 완료하지 못하였다.  
   -> 설치가 어려운 문제야 구글링해가면서 해결할 수 있겠지만, 최근 Point cloud에 대한 엄청나게 많은 연구가 이루어지고 있는데 이에 대한 업데이트가 이루어지지 않는다는 것이 큰 단점으로 작용할 것으로 보인다.
 
2. Open3D
   - Open3D는 2018년에 인텔에서 개발하여 배포하고 있는 라이브러리로, 최신 트렌드를 반영하고, 빠른 업데이트로 인해 많은 관심을 받고 있는 라이브러리라고 한다.
   - Open3D에서 제공하는 주요 기능은 아래와 같다.
     - 다양한 종류의 3차원 데이터를 위한 자료 구조(클래스) 제공
     - 데이터 종류마다 다른 표준 파일 포맷을 읽고 쓰는 기능
     - 2차원/3차원 데이터를 시각화(visualization)하는 기능
     - 2차원 영상 처리 알고리즘 (필터링 등)
     - 3차원 데이터를 위한 다양한 알고리즘
       - Local/Global Registration (ICP 등)
       - Normal Estimation
       - KD Tree
       - TSDF Volume Integration
       - Pose-graph Optimization
   - Open3D는 3rd Party 라이브러리에 대한 의존성이 최소화되어 있고, 내부 구조를 단순하게 만들어 설치 용량도 적고 설치도 매우 쉽게 할 수 있다는 장점이 있다.
   - 내부적으로는 C++로 구현하여 최적화 시켰지만, Python API로 바인딩되어 있어 Python에서 사용하기가 쉽고 Open3D 공식 documentation도 Python을 기반으로 설명되어 있다.
   - 또한 선형대수 계산 라이브러리인 numpy와도 연동이 잘 되어 행렬 연산이 매우 간편하다.
   - 마지막으로 OpenMP를 이용하여 계산량이 어마어마한 3D 데이터 처리를 병렬화하여 속도를 크게 향상시켰다고 한다.
3. Matlab
   - Matlab의 경우 툴박스 형태로 3D 영상처리 기능을 제공하고 있지만, 제공하는 기능이 제약적이고 유료라는 점도 단점으로 작용한다.


- Reference : 
  1. <https://pcl.gitbook.io/tutorial/part-0/part00-chapter00#3-2-3d>
  2. <https://goodgodgd.github.io/ian-flow/archivers/open3d-tutorial>  