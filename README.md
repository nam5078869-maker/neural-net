# 🧠 라이브러리 없이 C로 만든 신경망

외부 라이브러리 없이 **C 표준 라이브러리만으로** 신경망을 처음부터 구현했습니다.
행렬 연산부터 역전파, MNIST 손글씨 분류까지 모두 직접 만들었으며, **시험 데이터 정확도 97.76%** 를 달성했습니다.

PyTorch가 `loss.backward()` 한 줄로 처리하는 일을 직접 계산해 보면서, 딥러닝 프레임워크 내부에서 무슨 일이 일어나는지 이해하는 것이 목표였습니다.

## 구현한 것

| 단계 | 내용 |
|---|---|
| 1 | 행렬 연산 라이브러리 (곱, 전치, 원소별 연산) + 단위 테스트 14개 |
| 2 | 뉴런 하나로 AND/OR/NAND 학습, XOR이 불가능함을 확인 (선형 분리 불가) |
| 3 | 은닉층 + 역전파로 XOR 해결, **수치 미분으로 역전파 검증** |
| 4 | MNIST IDX 바이너리 파일 파서 (빅 엔디안 처리) |
| 5 | softmax + 교차 엔트로피, 미니배치 경사하강법으로 MNIST 학습 |
| 6 | 학습한 모델 저장/불러오기, 예측 프로그램과 혼동 행렬 |
| 7 | 반복문 순서에 따른 성능 비교, NumPy 구현과 속도 비교 |

## 결과

구조: `784 → 128 (sigmoid) → 10 (softmax)`, 가중치 101,770개
설정: 학습 6만 장, 배치 100, 학습률 0.5, 20에폭

```
에폭   손실      학습 정확도  시험 정확도
   1   0.47131      91.00%      91.88%
   5   0.15538      95.85%      95.48%
  10   0.09028      97.94%      97.04%
  15   0.06228      98.50%      97.37%
  20   0.04594      98.92%      97.76%

가장 높은 시험 정확도: 97.76%
```

`./predict`는 혼동 행렬로 어떤 숫자를 헷갈리는지 보여줍니다.

```
[혼동 행렬] 세로: 정답, 가로: 예측
          0    1    2    3    4    5    6    7    8    9   정확도
  0 |   951    .    1    1    .   10   10    4    2    1    97.0%
  4 |     1    4    4    .  898    .   12    3    3   57    91.4%
  5 |     7    5    2   43    9  778   13    7   19    9    87.2%
```

이미지 하나를 골라 예측 확률까지 확인할 수도 있습니다 (`./predict 8`). 틀릴 때도 97%로 확신하는 경우가 있어, 신경망이 자신의 오류를 알지 못한다는 점을 볼 수 있습니다.

## 실행 방법

필요한 것: C11 컴파일러(clang 또는 gcc), make, MNIST 데이터

```bash
# 1. MNIST 데이터 준비 (data 폴더에 압축을 푼 4개 파일)
mkdir -p data && cd data
curl -O https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/train-images-idx3-ubyte.gz
curl -O https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/train-labels-idx1-ubyte.gz
curl -O https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/t10k-images-idx3-ubyte.gz
curl -O https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/t10k-labels-idx1-ubyte.gz
gunzip *.gz && cd ..

# 2. 빌드
make

# 3. 실행
make test          # 행렬 라이브러리 테스트
make run-neuron    # 뉴런 하나로 논리 게이트 (XOR 실패 확인)
make run-xor       # 은닉층 + 역전파로 XOR 해결, 역전파 검증
make run-mnist     # MNIST 데이터 확인 (터미널에 손글씨 출력)
make run-bench     # 반복문 순서에 따른 행렬 곱 속도 비교

./mnist_train 60000 20 128 100 0.5   # 학습 (인자: 장수 에폭 은닉 배치 학습률)
./predict                            # 정확도 + 혼동 행렬
./predict 8 42 115                   # 특정 이미지 예측
```

## 프로젝트 구조

```
neural-net/
├── matrix.h / matrix.c        # 행렬 연산 (행 우선 1차원 배열)
├── test_matrix.c              # 행렬 라이브러리 단위 테스트
├── activation.h / activation.c# sigmoid와 그 미분
├── network.h / network.c      # 2층 신경망: 순전파, 역전파, 기울기 검증
├── mnist.h / mnist.c          # IDX 바이너리 파서
├── model.h / model.c          # 학습한 가중치 저장/불러오기
├── neuron.c                   # 2단계: 논리 게이트
├── xor.c                      # 3단계: XOR
├── mnist_demo.c               # 4단계: 데이터 확인
├── mnist_train.c              # 5단계: 학습
├── predict.c                  # 6단계: 예측 + 혼동 행렬
├── bench.c                    # 7단계: 성능 측정
├── numpy_train.py             # 7단계: 비교용 NumPy 구현
└── Makefile
```

## 수식과 구현

**순전파**

```
Z1 = X·W1 + b1     A1 = sigmoid(Z1)
Z2 = A1·W2 + b2    A2 = softmax(Z2)
```

**역전파** (연쇄법칙을 층마다 거꾸로 적용)

```
dZ2 = A2 - Y                    # 출력층 오차
dW2 = A1ᵀ·dZ2 / m,  db2 = mean(dZ2)
dA1 = dZ2·W2ᵀ                   # 오차를 은닉층으로 전달
dZ1 = dA1 ⊙ A1(1-A1)            # sigmoid 미분
dW1 = Xᵀ·dZ1 / m,   db1 = mean(dZ1)
```

출력층 식이 `A2 - Y`로 간단해지는 것은 활성화 함수와 손실 함수를 짝지어 썼기 때문입니다(sigmoid+이진 교차 엔트로피, softmax+교차 엔트로피). 그래서 출력층을 softmax로 바꿀 때 역전파 코드는 한 줄도 바뀌지 않았습니다.

### 역전파 검증

역전파는 부호 하나만 틀려도 조용히 잘못된 방향으로 학습합니다. 그래서 수치 미분과 비교하는 검증을 넣었습니다.

```
수치 기울기 ≈ ( L(w + h) - L(w - h) ) / (2h)
```

```
  W1  원소   8개, 최대 상대 오차 4.562e-10
  b1  원소   4개, 최대 상대 오차 2.673e-10
  W2  원소   4개, 최대 상대 오차 7.762e-11
  b2  원소   1개, 최대 상대 오차 2.595e-12
-> 역전파 구현 정상
```

## 성능

측정 환경: Intel Xeon 2.1GHz (2코어), gcc `-O2`, double 정밀도

**반복문 순서에 따른 차이** (`make run-bench`)

| 순서 | 시간 | 성능 |
|---|---|---|
| i-j-k (교과서식) | 1.848초 | 2.17 GFLOPS |
| i-k-j (이 구현) | 1.054초 | 3.81 GFLOPS |

결과는 완전히 같지만 **1.75배** 차이가 납니다. i-k-j 순서는 안쪽 반복에서 메모리를 순서대로 읽어 CPU 캐시를 잘 활용하기 때문입니다.

**NumPy와 비교** (학습 2만 장, 배치 100, 학습 루프만 측정)

| 구현 | 에폭당 시간 |
|---|---|
| C (이 구현) | 약 2.0초 |
| NumPy | 약 0.26초 |

**NumPy가 약 7배 빠릅니다.** NumPy의 행렬 곱은 내부적으로 BLAS를 호출하는데, BLAS는 SIMD 명령어, 캐시 블로킹, 멀티스레딩까지 적용된 수십 년간 최적화된 라이브러리입니다. "파이썬이라 느리다"는 통념과 달리, 무거운 계산이 C/포트란으로 된 라이브러리에서 처리되면 직접 짠 단순한 C 코드보다 빠를 수 있습니다.

이 프로젝트의 목적은 속도가 아니라 **동작 원리의 이해**였지만, 최적화 여지를 확인한 것도 수확이었습니다. 캐시 블로킹, SIMD(예: ARM NEON), OpenMP 병렬화를 적용하면 격차를 줄일 수 있습니다.

## 배운 점

- 순전파/역전파를 행렬 식으로 유도하고 코드로 옮기는 과정
- sigmoid+교차 엔트로피, softmax+교차 엔트로피에서 미분이 약분되는 이유
- 수치 미분을 이용한 역전파 검증 방법
- Xavier 초기화와 기울기 소실 문제
- 미니배치 경사하강법, 에폭마다 데이터를 섞는 이유
- 학습 정확도와 시험 정확도의 차이로 과적합 판단하기
- 바이너리 파일 파싱, 빅 엔디안 변환
- 메모리 관리: 스택 / `static` / 힙의 선택 기준, 중간 결과 행렬 재사용
- 반복문 순서와 캐시 지역성이 성능에 미치는 영향

## 개선 아이디어

- [ ] ReLU 활성화 함수와 He 초기화
- [ ] 은닉층을 여러 개 쌓을 수 있는 구조로 일반화
- [ ] 모멘텀, Adam 옵티마이저
- [ ] 드롭아웃, L2 정규화로 과적합 줄이기
- [ ] 합성곱(CNN) 구현으로 99% 도전
- [ ] 캐시 블로킹과 OpenMP로 행렬 곱 최적화
- [ ] 직접 그린 숫자 이미지를 입력해 예측하기

## 개발 환경

- macOS, Visual Studio Code
- Apple clang, C11
- 외부 라이브러리 없음 (비교용 `numpy_train.py` 제외)
