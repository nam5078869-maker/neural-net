#ifndef MATRIX_H
#define MATRIX_H

/*
 * 신경망에 필요한 기본 행렬 연산.
 *
 * 2차원 행렬을 1차원 배열 하나에 행 우선(row-major)으로 저장한다.
 *   (i행, j열) 값 = data[i * cols + j]
 *
 * 결과를 담을 행렬(out)을 인자로 받는 함수가 많다.
 * 학습 중에는 같은 연산을 수만 번 반복하므로,
 * 매번 새 메모리를 할당하지 않고 미리 만든 행렬을 재사용하기 위해서다.
 */

typedef struct {
    int     rows;
    int     cols;
    double *data;
} Matrix;

/* (i, j) 원소에 접근하는 매크로. 읽기와 쓰기 모두 가능 */
#define MAT_AT(m, i, j) ((m)->data[(i) * (m)->cols + (j)])

/* ---------- 생성과 해제 ---------- */

/* rows x cols 크기의 행렬을 만들고 0으로 채운다. 실패하면 NULL */
Matrix *mat_create(int rows, int cols);

/* 행렬 메모리를 해제한다. NULL이면 아무것도 하지 않는다 */
void mat_free(Matrix *m);

/* 1차원 배열 값으로 채운다. values 길이는 rows * cols 이어야 한다 */
void mat_set_values(Matrix *m, const double *values);

/* 모든 원소를 value로 채운다 */
void mat_fill(Matrix *m, double value);

/* 모든 원소를 [-limit, limit] 범위의 균등분포 난수로 채운다 */
void mat_randomize(Matrix *m, double limit);

/* src의 값을 dst로 복사한다 (크기가 같아야 함) */
void mat_copy(Matrix *dst, const Matrix *src);

/* ---------- 연산 (결과는 out에 저장) ---------- */

/* 행렬 곱: out = a * b   (a.cols == b.rows, out은 a.rows x b.cols) */
void mat_mul(Matrix *out, const Matrix *a, const Matrix *b);

/* 원소별 덧셈: out = a + b */
void mat_add(Matrix *out, const Matrix *a, const Matrix *b);

/* 원소별 뺄셈: out = a - b */
void mat_sub(Matrix *out, const Matrix *a, const Matrix *b);

/* 원소별 곱 (아다마르 곱): out = a ⊙ b */
void mat_hadamard(Matrix *out, const Matrix *a, const Matrix *b);

/* 스칼라 곱: out = a * k */
void mat_scale(Matrix *out, const Matrix *a, double k);

/* 전치 행렬: out = aᵀ   (out은 a.cols x a.rows) */
void mat_transpose(Matrix *out, const Matrix *a);

/* 모든 원소에 함수를 적용: out[i] = f(a[i]) */
void mat_apply(Matrix *out, const Matrix *a, double (*f)(double));

/* ---------- 기타 ---------- */

/* 두 행렬의 모든 원소 차이가 eps 이하인지 (테스트용). 같으면 1 */
int mat_equals(const Matrix *a, const Matrix *b, double eps);

/* 행렬을 보기 좋게 출력한다 */
void mat_print(const Matrix *m, const char *name);

#endif
