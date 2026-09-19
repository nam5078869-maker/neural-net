/*
 * 2단계: 뉴런 하나로 논리 게이트(AND, OR, NAND, XOR) 학습하기
 *
 *   입력 x1, x2  ──▶  z = w1*x1 + w2*x2 + b  ──▶  a = sigmoid(z)  ──▶  출력(0~1)
 *
 * 학습 방법: 경사하강법
 *   1) 순전파: 현재 가중치로 출력 a를 계산
 *   2) 손실: 정답 y와 얼마나 다른지 계산 (이진 교차 엔트로피)
 *   3) 기울기: 손실을 줄이려면 w, b를 어느 방향으로 바꿔야 하는지 계산
 *   4) 갱신: w = w - 학습률 * 기울기
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "activation.h"
#include "matrix.h"

#define N_SAMPLES     4       /* 입력 조합 4가지 */
#define N_INPUTS      2       /* x1, x2 */
#define EPOCHS        5000    /* 전체 데이터를 몇 번 반복해서 학습할지 */
#define LEARNING_RATE 0.5     /* 한 번에 가중치를 얼마나 움직일지 */
#define PRINT_EVERY   1000
#define RANDOM_SEED   42      /* 같은 결과를 재현하기 위해 고정 */

/* 모든 게이트가 공통으로 쓰는 입력 (4행 x 2열) */
static const double INPUTS[N_SAMPLES * N_INPUTS] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1,
};

typedef struct {
    const char *name;
    double      targets[N_SAMPLES];   /* 각 입력에 대한 정답 */
} Gate;

static const Gate GATES[] = {
    { "AND",  { 0, 0, 0, 1 } },
    { "OR",   { 0, 1, 1, 1 } },
    { "NAND", { 1, 1, 1, 0 } },
    { "XOR",  { 0, 1, 1, 0 } },
};
#define N_GATES (int)(sizeof(GATES) / sizeof(GATES[0]))

/* 뉴런 하나의 학습 파라미터 */
typedef struct {
    Matrix *W;     /* 가중치 (1 x N_INPUTS) */
    double  b;     /* 편향 */
} Neuron;

/* 계산 중간값을 담는 행렬들. 매 반복마다 새로 만들지 않고 재사용한다 */
typedef struct {
    Matrix *Wt;    /* Wᵀ (N_INPUTS x 1) */
    Matrix *Z;     /* z = X·Wᵀ + b (N_SAMPLES x 1) */
    Matrix *A;     /* a = sigmoid(z) (N_SAMPLES x 1) */
    Matrix *dZ;    /* 손실을 z로 미분한 값 (N_SAMPLES x 1) */
    Matrix *dZt;   /* dZᵀ (1 x N_SAMPLES) */
    Matrix *dW;    /* 손실을 W로 미분한 값 (1 x N_INPUTS) */
} Workspace;

/* ---------------- 순전파 ---------------- */

/* 4개 입력을 한꺼번에 계산한다: A = sigmoid(X·Wᵀ + b) */
static void forward(const Neuron *n, const Matrix *X, Workspace *ws) {
    mat_transpose(ws->Wt, n->W);
    mat_mul(ws->Z, X, ws->Wt);                 /* 행마다 w1*x1 + w2*x2 */
    for (int i = 0; i < ws->Z->rows; i++) {
        MAT_AT(ws->Z, i, 0) += n->b;           /* + b */
    }
    mat_apply(ws->A, ws->Z, sigmoid);
}

/* ---------------- 손실 ---------------- */

/* 이진 교차 엔트로피: -평균[ y*log(a) + (1-y)*log(1-a) ]
 * 정답에 가까울수록 0에 가까워지고, 틀릴수록 커진다. */
static double binary_cross_entropy(const Matrix *A, const Matrix *Y) {
    const double eps = 1e-12;                  /* log(0) 방지 */
    double sum = 0.0;
    for (int i = 0; i < A->rows; i++) {
        double a = MAT_AT(A, i, 0);
        double y = MAT_AT(Y, i, 0);
        sum += y * log(a + eps) + (1.0 - y) * log(1.0 - a + eps);
    }
    return -sum / A->rows;
}

/* ---------------- 역전파 + 갱신 ---------------- */

/*
 * 시그모이드 + 이진 교차 엔트로피 조합에서는 미분이 간단해진다.
 *   dL/dz = a - y
 *   dL/dW = (dZᵀ · X) / m      (m = 샘플 수)
 *   dL/db = 평균(dZ)
 */
static void backward_and_update(Neuron *n, const Matrix *X, const Matrix *Y,
                                Workspace *ws, double learning_rate) {
    int m = X->rows;

    mat_sub(ws->dZ, ws->A, Y);                 /* dZ = A - Y */

    mat_transpose(ws->dZt, ws->dZ);
    mat_mul(ws->dW, ws->dZt, X);               /* dW = dZᵀ · X */
    mat_scale(ws->dW, ws->dW, 1.0 / m);        /*      / m     */

    double db = 0.0;
    for (int i = 0; i < m; i++) {
        db += MAT_AT(ws->dZ, i, 0);
    }
    db /= m;

    /* 기울기의 반대 방향으로 조금 이동 */
    mat_scale(ws->dW, ws->dW, learning_rate);
    mat_sub(n->W, n->W, ws->dW);
    n->b -= learning_rate * db;
}

/* ---------------- 게이트 하나 학습 ---------------- */

/* 학습 후 맞힌 개수를 돌려준다 */
static int train_gate(const Gate *gate) {
    Matrix *X = mat_create(N_SAMPLES, N_INPUTS);
    Matrix *Y = mat_create(N_SAMPLES, 1);
    mat_set_values(X, INPUTS);
    mat_set_values(Y, gate->targets);

    Neuron n = { mat_create(1, N_INPUTS), 0.0 };
    mat_randomize(n.W, 1.0);

    Workspace ws = {
        .Wt  = mat_create(N_INPUTS, 1),
        .Z   = mat_create(N_SAMPLES, 1),
        .A   = mat_create(N_SAMPLES, 1),
        .dZ  = mat_create(N_SAMPLES, 1),
        .dZt = mat_create(1, N_SAMPLES),
        .dW  = mat_create(1, N_INPUTS),
    };

    printf("\n==================== %s ====================\n", gate->name);

    for (int epoch = 0; epoch <= EPOCHS; epoch++) {
        forward(&n, X, &ws);

        if (epoch % PRINT_EVERY == 0) {
            printf("epoch %5d   loss %.6f\n",
                   epoch, binary_cross_entropy(ws.A, Y));
        }
        if (epoch == EPOCHS) {
            break;                 /* 마지막은 결과 확인용 순전파만 */
        }

        backward_and_update(&n, X, Y, &ws, LEARNING_RATE);
    }

    /* 결과 표 */
    printf("\n x1  x2 | 정답 | 출력    | 예측\n");
    printf("--------+------+---------+---------\n");
    int correct = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double a = MAT_AT(ws.A, i, 0);
        int predicted = a >= 0.5 ? 1 : 0;
        int target = (int)MAT_AT(Y, i, 0);
        int ok = predicted == target;
        correct += ok;
        printf("  %d   %d |   %d  | %.4f  |  %d  %s\n",
               (int)MAT_AT(X, i, 0), (int)MAT_AT(X, i, 1),
               target, a, predicted, ok ? "O" : "X");
    }
    printf("\n학습된 값: w1 = %.3f, w2 = %.3f, b = %.3f\n",
           MAT_AT(n.W, 0, 0), MAT_AT(n.W, 0, 1), n.b);
    printf("정확도: %d / %d\n", correct, N_SAMPLES);

    mat_free(X);
    mat_free(Y);
    mat_free(n.W);
    mat_free(ws.Wt);
    mat_free(ws.Z);
    mat_free(ws.A);
    mat_free(ws.dZ);
    mat_free(ws.dZt);
    mat_free(ws.dW);
    return correct;
}

int main(void) {
    srand(RANDOM_SEED);

    int results[N_GATES];
    for (int g = 0; g < N_GATES; g++) {
        results[g] = train_gate(&GATES[g]);
    }

    printf("\n==================== 요약 ====================\n");
    for (int g = 0; g < N_GATES; g++) {
        printf("%-5s %d / %d  %s\n", GATES[g].name, results[g], N_SAMPLES,
               results[g] == N_SAMPLES ? "학습 성공" : "학습 실패");
    }
    return 0;
}
