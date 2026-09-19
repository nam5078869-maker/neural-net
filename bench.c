/*
 * 7단계: 행렬 곱 성능 측정
 *
 * 1단계에서 반복문을 i-k-j 순서로 쓴 이유를 직접 확인한다.
 * 계산 결과는 같지만 메모리를 읽는 순서가 달라 속도 차이가 난다.
 *
 * 실행: ./bench
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"

/* 학습에서 실제로 쓰는 크기: (배치 100 x 입력 784) x (784 x 은닉 128) */
#define M 100
#define K 784
#define N 128
#define REPEAT 200

/* 흔히 쓰는 i-j-k 순서: 안쪽에서 b를 세로로 읽어 캐시 효율이 나쁘다 */
static void mul_ijk(Matrix *out, const Matrix *a, const Matrix *b) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < a->cols; k++) {
                sum += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
            MAT_AT(out, i, j) = sum;
        }
    }
}

static double elapsed(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

/* 이 곱셈의 연산 횟수: 곱셈과 덧셈 각각 M*K*N번 */
static double gflops(double seconds) {
    double ops = 2.0 * M * K * N * REPEAT;
    return ops / seconds / 1e9;
}

int main(void) {
    srand(42);

    Matrix *a   = mat_create(M, K);
    Matrix *b   = mat_create(K, N);
    Matrix *out = mat_create(M, N);
    Matrix *ref = mat_create(M, N);
    if (!a || !b || !out || !ref) {
        return 1;
    }
    mat_randomize(a, 1.0);
    mat_randomize(b, 1.0);

    printf("행렬 곱 (%d x %d) x (%d x %d), %d회 반복\n\n", M, K, K, N, REPEAT);

    clock_t t = clock();
    for (int r = 0; r < REPEAT; r++) {
        mul_ijk(out, a, b);
    }
    double time_ijk = elapsed(t);
    mat_copy(ref, out);

    t = clock();
    for (int r = 0; r < REPEAT; r++) {
        mat_mul(out, a, b);      /* 우리 구현: i-k-j */
    }
    double time_ikj = elapsed(t);

    printf("i-j-k 순서 (교과서식): %6.3f초  %5.2f GFLOPS\n",
           time_ijk, gflops(time_ijk));
    printf("i-k-j 순서 (우리 것):  %6.3f초  %5.2f GFLOPS\n",
           time_ikj, gflops(time_ikj));
    printf("\n같은 결과인가: %s\n",
           mat_equals(ref, out, 1e-9) ? "예" : "아니오");
    printf("속도 차이: %.2f배\n", time_ijk / time_ikj);

    mat_free(a);
    mat_free(b);
    mat_free(out);
    mat_free(ref);
    return 0;
}
