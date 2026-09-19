CC      = clang
# -O2: 최적화. 학습 속도가 몇 배 차이 난다
CFLAGS  = -Wall -Wextra -std=c11 -O2 -g
LDLIBS  = -lm

LIB_OBJS = matrix.o activation.o network.o mnist.o model.o
PROGRAMS = test_matrix neuron xor mnist_demo mnist_train predict bench

all: $(PROGRAMS)

test_matrix: test_matrix.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

neuron: neuron.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

xor: xor.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

mnist_demo: mnist_demo.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

mnist_train: mnist_train.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

predict: predict.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

bench: bench.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c matrix.h activation.h network.h mnist.h model.h
	$(CC) $(CFLAGS) -c $<

# 행렬 테스트
test: test_matrix
	./test_matrix

# 2단계: 뉴런 하나로 논리 게이트
run-neuron: neuron
	./neuron

# 3단계: 은닉층으로 XOR
run-xor: xor
	./xor

# 4단계: MNIST 데이터 확인
run-mnist: mnist_demo
	./mnist_demo

# 5단계: MNIST 학습
train: mnist_train
	./mnist_train

# 6단계: 저장한 모델로 예측 (전체 정확도와 혼동 행렬)
run-predict: predict
	./predict

# 7단계: 반복문 순서에 따른 행렬 곱 속도 비교
run-bench: bench
	./bench

# 7단계: NumPy 구현과 비교 (numpy 필요: pip3 install numpy)
run-numpy:
	python3 numpy_train.py 20000 3

# 메모리 오류 검사를 켜고 빌드 (느리므로 학습에는 쓰지 말 것)
debug: CFLAGS := -Wall -Wextra -std=c11 -g -fsanitize=address,undefined
debug: clean all

clean:
	rm -f *.o $(PROGRAMS)

.PHONY: all test run-neuron run-xor run-mnist train run-predict run-bench run-numpy debug clean
