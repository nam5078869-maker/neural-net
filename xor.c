/*
 * 3단계: 은닉층을 추가해 XOR 학습하기
 *
 * 2단계에서 뉴런 하나로는 XOR을 풀 수 없었다(손실 0.693에서 멈춤).
 * 은닉층을 넣으면 풀린다는 것을 확인하고, 역전파가 맞는지도 검증한다.
 */
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "network.h"

#define N_SAMPLES     4
#define N_INPUTS      2
#define N_HIDDEN      4
#define N_OUTPUTS     1
#define EPOCHS        20000
#define LEARNING_RATE 0.5
#define PRINT_EVERY   2000
#define RANDOM_SEED   42

static const double INPUTS[N_SAMPLES * N_INPUTS] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1,
};
static const double XOR_TARGETS[N_SAMPLES] = { 0, 1, 1, 0 };

int main(void) {
    srand(RANDOM_SEED);

    Matrix *X = mat_create(N_SAMPLES, N_INPUTS);
    Matrix *Y = mat_create(N_SAMPLES, N_OUTPUTS);
    mat_set_values(X, INPUTS);
    mat_set_values(Y, XOR_TARGETS);

    Network *net = net_create(N_INPUTS, N_HIDDEN, N_OUTPUTS, N_SAMPLES);
    if (net == NULL) {
        printf("신경망을 만들 수 없습니다 (메모리 부족)\n");
        return 1;
    }

    printf("구조: 입력 %d -> 은닉 %d -> 출력 %d\n",
           N_INPUTS, N_HIDDEN, N_OUTPUTS);

    /* ---- 학습 전에 역전파 검증 ---- */
    printf("\n[역전파 검증: 수치 미분과 비교]\n");
    double worst = net_gradient_check(net, X, Y);
    printf("전체 최대 상대 오차: %.3e  %s\n", worst,
           worst < 1e-6 ? "-> 역전파 구현 정상" : "-> 구현을 다시 확인하세요");

    /* ---- 학습 ---- */
    printf("\n[학습]\n");
    for (int epoch = 0; epoch <= EPOCHS; epoch++) {
        if (epoch % PRINT_EVERY == 0) {
            net_forward(net, X);
            printf("epoch %6d   loss %.6f\n", epoch, net_loss(net, Y));
        }
        if (epoch == EPOCHS) {
            break;
        }
        net_train_step(net, X, Y, LEARNING_RATE);
    }

    /* ---- 결과 ---- */
    net_forward(net, X);
    printf("\n x1  x2 | 정답 | 출력    | 예측\n");
    printf("--------+------+---------+---------\n");
    int correct = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double a = MAT_AT(net->A2, i, 0);
        int predicted = a >= 0.5 ? 1 : 0;
        int target = (int)MAT_AT(Y, i, 0);
        int ok = predicted == target;
        correct += ok;
        printf("  %d   %d |   %d  | %.4f  |  %d  %s\n",
               (int)MAT_AT(X, i, 0), (int)MAT_AT(X, i, 1),
               target, a, predicted, ok ? "O" : "X");
    }
    printf("\n정확도: %d / %d\n", correct, N_SAMPLES);

    /* ---- 은닉층이 무엇을 배웠는지 ---- */
    printf("\n[은닉층 출력] 각 입력에 대해 은닉 뉴런 %d개가 내는 값\n", N_HIDDEN);
    printf(" x1  x2 | ");
    for (int j = 0; j < N_HIDDEN; j++) {
        printf("h%d     ", j + 1);
    }
    printf("\n--------+");
    for (int j = 0; j < N_HIDDEN; j++) {
        printf("-------");
    }
    printf("\n");
    for (int i = 0; i < N_SAMPLES; i++) {
        printf("  %d   %d | ", (int)MAT_AT(X, i, 0), (int)MAT_AT(X, i, 1));
        for (int j = 0; j < N_HIDDEN; j++) {
            printf("%.3f  ", MAT_AT(net->A1, i, j));
        }
        printf("\n");
    }

    net_free(net);
    mat_free(X);
    mat_free(Y);
    return 0;
}
