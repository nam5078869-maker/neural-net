/*
 * 행렬 라이브러리 테스트.
 * 손으로 계산한 정답과 비교해서 모두 맞으면 "모든 테스트 통과"를 출력한다.
 */
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

#define EPS 1e-9

static int tests_run = 0;
static int tests_failed = 0;

/* 조건이 거짓이면 실패로 기록하고 어디서 실패했는지 출력 */
#define CHECK(cond, msg)                                              \
    do {                                                              \
        tests_run++;                                                  \
        if (!(cond)) {                                                \
            tests_failed++;                                           \
            printf("  [실패] %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        } else {                                                      \
            printf("  [통과] %s\n", msg);                             \
        }                                                             \
    } while (0)

static double square(double x) {
    return x * x;
}

static void test_create(void) {
    Matrix *m = mat_create(2, 3);
    CHECK(m != NULL, "행렬 생성");
    CHECK(m->rows == 2 && m->cols == 3, "크기 2x3");

    int all_zero = 1;
    for (int i = 0; i < 6; i++) {
        if (m->data[i] != 0.0) all_zero = 0;
    }
    CHECK(all_zero, "생성 직후 모든 원소가 0");

    MAT_AT(m, 1, 2) = 7.0;
    CHECK(m->data[1 * 3 + 2] == 7.0, "MAT_AT(1,2) = data[5] (행 우선 저장)");

    mat_free(m);
}

static void test_mul(void) {
    /*
     * [1 2 3]   [ 7  8]   [ 58  64]
     * [4 5 6] x [ 9 10] = [139 154]
     *           [11 12]
     */
    Matrix *a = mat_create(2, 3);
    Matrix *b = mat_create(3, 2);
    Matrix *out = mat_create(2, 2);
    Matrix *expected = mat_create(2, 2);

    mat_set_values(a, (double[]){1, 2, 3, 4, 5, 6});
    mat_set_values(b, (double[]){7, 8, 9, 10, 11, 12});
    mat_set_values(expected, (double[]){58, 64, 139, 154});

    mat_mul(out, a, b);
    CHECK(mat_equals(out, expected, EPS), "행렬 곱 (2x3) x (3x2)");

    mat_print(a, "a");
    mat_print(b, "b");
    mat_print(out, "a x b");

    mat_free(a);
    mat_free(b);
    mat_free(out);
    mat_free(expected);
}

static void test_elementwise(void) {
    Matrix *a = mat_create(2, 2);
    Matrix *b = mat_create(2, 2);
    Matrix *out = mat_create(2, 2);
    Matrix *expected = mat_create(2, 2);

    mat_set_values(a, (double[]){1, 2, 3, 4});
    mat_set_values(b, (double[]){10, 20, 30, 40});

    mat_add(out, a, b);
    mat_set_values(expected, (double[]){11, 22, 33, 44});
    CHECK(mat_equals(out, expected, EPS), "덧셈");

    mat_sub(out, b, a);
    mat_set_values(expected, (double[]){9, 18, 27, 36});
    CHECK(mat_equals(out, expected, EPS), "뺄셈");

    mat_hadamard(out, a, b);
    mat_set_values(expected, (double[]){10, 40, 90, 160});
    CHECK(mat_equals(out, expected, EPS), "원소별 곱");

    mat_scale(out, a, 0.5);
    mat_set_values(expected, (double[]){0.5, 1, 1.5, 2});
    CHECK(mat_equals(out, expected, EPS), "스칼라 곱");

    mat_apply(out, a, square);
    mat_set_values(expected, (double[]){1, 4, 9, 16});
    CHECK(mat_equals(out, expected, EPS), "함수 적용 (제곱)");

    /* 결과를 입력 행렬에 바로 저장해도 되는지 (원소별 연산은 가능) */
    mat_add(a, a, a);
    mat_set_values(expected, (double[]){2, 4, 6, 8});
    CHECK(mat_equals(a, expected, EPS), "제자리 연산 a = a + a");

    mat_free(a);
    mat_free(b);
    mat_free(out);
    mat_free(expected);
}

static void test_transpose(void) {
    Matrix *a = mat_create(2, 3);
    Matrix *out = mat_create(3, 2);
    Matrix *expected = mat_create(3, 2);

    mat_set_values(a, (double[]){1, 2, 3, 4, 5, 6});
    mat_set_values(expected, (double[]){1, 4, 2, 5, 3, 6});

    mat_transpose(out, a);
    CHECK(mat_equals(out, expected, EPS), "전치 (2x3 -> 3x2)");

    mat_free(a);
    mat_free(out);
    mat_free(expected);
}

static void test_randomize(void) {
    Matrix *m = mat_create(100, 100);
    srand(42);
    mat_randomize(m, 0.5);

    int in_range = 1;
    double sum = 0.0;
    for (int i = 0; i < 10000; i++) {
        if (m->data[i] < -0.5 || m->data[i] > 0.5) in_range = 0;
        sum += m->data[i];
    }
    CHECK(in_range, "난수가 모두 [-0.5, 0.5] 범위");
    CHECK(sum / 10000 > -0.05 && sum / 10000 < 0.05, "난수 평균이 0 근처");

    mat_free(m);
}

int main(void) {
    printf("[생성]\n");        test_create();
    printf("\n[행렬 곱]\n");   test_mul();
    printf("\n[원소별 연산]\n"); test_elementwise();
    printf("\n[전치]\n");      test_transpose();
    printf("\n[난수]\n");      test_randomize();

    printf("\n%d개 중 %d개 통과\n", tests_run, tests_run - tests_failed);
    if (tests_failed == 0) {
        printf("모든 테스트 통과\n");
        return 0;
    }
    return 1;
}
