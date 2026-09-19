#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"

/*
 * 은닉층이 하나 있는 2층 신경망.
 *
 *   입력 X        은닉층 A1              출력 A2
 *  (m x n_in) → (m x n_hidden) →  (m x n_out)
 *
 *   Z1 = X·W1 + b1,   A1 = sigmoid(Z1)
 *   Z2 = A1·W2 + b2,  A2 = sigmoid(Z2)
 *
 * m은 한 번에 처리하는 샘플 수(배치 크기)다.
 */

/* 출력층에서 쓸 함수 */
typedef enum {
    OUT_SIGMOID,   /* 0/1 판단 (XOR처럼 출력이 1개일 때) */
    OUT_SOFTMAX    /* 여러 개 중 하나 고르기 (숫자 0~9 분류) */
} OutputActivation;

typedef struct {
    OutputActivation output;
    int n_in;
    int n_hidden;
    int n_out;
    int batch;

    /* 학습 대상 (파라미터) */
    Matrix *W1;   /* n_in x n_hidden     */
    Matrix *b1;   /* 1 x n_hidden        */
    Matrix *W2;   /* n_hidden x n_out    */
    Matrix *b2;   /* 1 x n_out           */

    /* 순전파 중간값 (역전파에서 다시 사용한다) */
    Matrix *Z1, *A1, *Z2, *A2;         /* batch x ... */

    /* 역전파 기울기 */
    Matrix *dZ1, *dW1, *db1;
    Matrix *dZ2, *dW2, *db2;

    /* 계산용 임시 행렬 */
    Matrix *A1t, *Xt, *W2t, *dA1, *A1grad;
} Network;

/* 신경망을 만들고 가중치를 무작위로 초기화한다. 실패하면 NULL */
Network *net_create(int n_in, int n_hidden, int n_out, int batch);

/* 출력층을 softmax로 바꾼다 (기본값은 sigmoid).
 * 손실도 자동으로 다중 분류용 교차 엔트로피로 바뀐다. */
void net_use_softmax(Network *net, int on);

/* 신경망 메모리를 해제한다 */
void net_free(Network *net);

/* 순전파. 결과는 net->A2에 담긴다 (X: batch x n_in) */
void net_forward(Network *net, const Matrix *X);

/* 손실. sigmoid면 이진 교차 엔트로피, softmax면 다중 분류 교차 엔트로피.
 * net_forward를 먼저 호출해야 한다 */
double net_loss(const Network *net, const Matrix *Y);

/* 역전파. 기울기를 net->dW1, dW2, db1, db2에 계산한다 */
void net_backward(Network *net, const Matrix *X, const Matrix *Y);

/* 계산된 기울기로 가중치를 갱신한다 */
void net_update(Network *net, double learning_rate);

/* 순전파 → 손실 → 역전파 → 갱신을 한 번 수행하고 손실을 돌려준다 */
double net_train_step(Network *net, const Matrix *X, const Matrix *Y,
                      double learning_rate);

/* 역전파가 맞는지 수치 미분과 비교해 검증한다.
 * 가장 큰 상대 오차를 돌려준다 (1e-6 이하면 정상) */
double net_gradient_check(Network *net, const Matrix *X, const Matrix *Y);

#endif
