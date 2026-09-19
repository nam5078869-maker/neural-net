#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "activation.h"
#include "matrix.h"
#include "network.h"

/* out = A·B + bias  (bias는 1 x cols, 모든 행에 같은 값을 더한다) */
static void affine(Matrix *out, const Matrix *A, const Matrix *B,
                   const Matrix *bias) {
    mat_mul(out, A, B);
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) += MAT_AT(bias, 0, j);
        }
    }
}

/* 열마다 평균을 구한다: out(1 x cols) = mean(rows of m) */
static void column_mean(Matrix *out, const Matrix *m) {
    assert(out->rows == 1 && out->cols == m->cols);
    for (int j = 0; j < m->cols; j++) {
        double sum = 0.0;
        for (int i = 0; i < m->rows; i++) {
            sum += MAT_AT(m, i, j);
        }
        MAT_AT(out, 0, j) = sum / m->rows;
    }
}

/* param = param - k * grad  (가중치 갱신용) */
static void step_down(Matrix *param, const Matrix *grad, double k) {
    for (int i = 0; i < param->rows * param->cols; i++) {
        param->data[i] -= k * grad->data[i];
    }
}

/* ================= 생성과 해제 ================= */

/*
 * Xavier(Glorot) 초기화: 가중치 범위를 층 크기에 맞춰 정한다.
 *   limit = sqrt(6 / (입력 수 + 출력 수))
 * 너무 크면 시그모이드가 0이나 1에 붙어 기울기가 사라지고,
 * 너무 작으면 학습이 거의 진행되지 않는다.
 */
static void init_weights(Matrix *W, int n_in, int n_out) {
    double limit = sqrt(6.0 / (n_in + n_out));
    mat_randomize(W, limit);
}

Network *net_create(int n_in, int n_hidden, int n_out, int batch) {
    Network *net = calloc(1, sizeof(Network));
    if (net == NULL) {
        return NULL;
    }

    net->n_in     = n_in;
    net->n_hidden = n_hidden;
    net->n_out    = n_out;
    net->batch    = batch;

    net->W1 = mat_create(n_in, n_hidden);
    net->b1 = mat_create(1, n_hidden);
    net->W2 = mat_create(n_hidden, n_out);
    net->b2 = mat_create(1, n_out);

    net->Z1 = mat_create(batch, n_hidden);
    net->A1 = mat_create(batch, n_hidden);
    net->Z2 = mat_create(batch, n_out);
    net->A2 = mat_create(batch, n_out);

    net->dZ1 = mat_create(batch, n_hidden);
    net->dW1 = mat_create(n_in, n_hidden);
    net->db1 = mat_create(1, n_hidden);
    net->dZ2 = mat_create(batch, n_out);
    net->dW2 = mat_create(n_hidden, n_out);
    net->db2 = mat_create(1, n_out);

    net->A1t    = mat_create(n_hidden, batch);
    net->Xt     = mat_create(n_in, batch);
    net->W2t    = mat_create(n_out, n_hidden);
    net->dA1    = mat_create(batch, n_hidden);
    net->A1grad = mat_create(batch, n_hidden);

    /* 하나라도 할당에 실패하면 전부 해제하고 NULL */
    Matrix *all[] = { net->W1, net->b1, net->W2, net->b2,
                      net->Z1, net->A1, net->Z2, net->A2,
                      net->dZ1, net->dW1, net->db1,
                      net->dZ2, net->dW2, net->db2,
                      net->A1t, net->Xt, net->W2t, net->dA1, net->A1grad };
    for (int i = 0; i < (int)(sizeof(all) / sizeof(all[0])); i++) {
        if (all[i] == NULL) {
            net_free(net);
            return NULL;
        }
    }

    init_weights(net->W1, n_in, n_hidden);
    init_weights(net->W2, n_hidden, n_out);
    /* 편향은 0에서 시작하는 것이 일반적이다 (mat_create가 0으로 채운다) */

    return net;
}

void net_free(Network *net) {
    if (net == NULL) {
        return;
    }
    mat_free(net->W1);  mat_free(net->b1);
    mat_free(net->W2);  mat_free(net->b2);
    mat_free(net->Z1);  mat_free(net->A1);
    mat_free(net->Z2);  mat_free(net->A2);
    mat_free(net->dZ1); mat_free(net->dW1); mat_free(net->db1);
    mat_free(net->dZ2); mat_free(net->dW2); mat_free(net->db2);
    mat_free(net->A1t); mat_free(net->Xt);  mat_free(net->W2t);
    mat_free(net->dA1); mat_free(net->A1grad);
    free(net);
}

void net_use_softmax(Network *net, int on) {
    net->output = on ? OUT_SOFTMAX : OUT_SIGMOID;
}

/* ================= 순전파 ================= */

/*
 * softmax: 행마다 값들을 "합이 1인 확률"로 바꾼다.
 *
 *   softmax(z)_k = e^(z_k) / Σ e^(z_j)
 *
 * 그대로 계산하면 e^(큰 수)가 무한대가 되어버린다(오버플로).
 * 모든 값에서 최댓값을 빼도 결과가 같다는 성질을 이용해 이를 막는다.
 */
static void softmax_rows(Matrix *out, const Matrix *Z) {
    for (int i = 0; i < Z->rows; i++) {
        double max = MAT_AT(Z, i, 0);
        for (int j = 1; j < Z->cols; j++) {
            if (MAT_AT(Z, i, j) > max) {
                max = MAT_AT(Z, i, j);
            }
        }

        double sum = 0.0;
        for (int j = 0; j < Z->cols; j++) {
            double e = exp(MAT_AT(Z, i, j) - max);
            MAT_AT(out, i, j) = e;
            sum += e;
        }
        for (int j = 0; j < Z->cols; j++) {
            MAT_AT(out, i, j) /= sum;
        }
    }
}

void net_forward(Network *net, const Matrix *X) {
    assert(X->rows == net->batch && X->cols == net->n_in);

    affine(net->Z1, X, net->W1, net->b1);          /* Z1 = X·W1 + b1  */
    mat_apply(net->A1, net->Z1, sigmoid);          /* A1 = sigmoid(Z1) */

    affine(net->Z2, net->A1, net->W2, net->b2);    /* Z2 = A1·W2 + b2 */

    if (net->output == OUT_SOFTMAX) {
        softmax_rows(net->A2, net->Z2);
    } else {
        mat_apply(net->A2, net->Z2, sigmoid);
    }
}

double net_loss(const Network *net, const Matrix *Y) {
    const double eps = 1e-12;
    double sum = 0.0;
    int n = net->A2->rows * net->A2->cols;

    if (net->output == OUT_SOFTMAX) {
        /* 다중 분류 교차 엔트로피: 정답 자리의 확률만 본다.
         * Y가 원-핫이라 정답이 아닌 자리는 y=0이 되어 사라진다. */
        for (int i = 0; i < n; i++) {
            sum += Y->data[i] * log(net->A2->data[i] + eps);
        }
    } else {
        for (int i = 0; i < n; i++) {
            double a = net->A2->data[i];
            double y = Y->data[i];
            sum += y * log(a + eps) + (1.0 - y) * log(1.0 - a + eps);
        }
    }
    return -sum / net->A2->rows;
}

/* ================= 역전파 ================= */

/*
 * 연쇄법칙을 층마다 거꾸로 적용한다.
 *
 * 놀랍게도 출력층 식은 sigmoid든 softmax든 똑같이 dZ2 = A2 - Y 다.
 * 각 활성화 함수에 짝이 맞는 손실 함수를 쓰면 미분이 이렇게 약분된다.
 *
 *   출력층:  dZ2 = A2 - Y                       (예측 - 정답)
 *            dW2 = A1ᵀ·dZ2 / m,  db2 = mean(dZ2)
 *
 *   은닉층:  dA1 = dZ2·W2ᵀ                      (오차를 뒤로 전달)
 *            dZ1 = dA1 ⊙ A1(1-A1)               (시그모이드 미분)
 *            dW1 = Xᵀ·dZ1 / m,   db1 = mean(dZ1)
 */
void net_backward(Network *net, const Matrix *X, const Matrix *Y) {
    double inv_m = 1.0 / net->batch;

    /* ---- 출력층 ---- */
    mat_sub(net->dZ2, net->A2, Y);

    mat_transpose(net->A1t, net->A1);
    mat_mul(net->dW2, net->A1t, net->dZ2);
    mat_scale(net->dW2, net->dW2, inv_m);
    column_mean(net->db2, net->dZ2);

    /* ---- 은닉층 ---- */
    mat_transpose(net->W2t, net->W2);
    mat_mul(net->dA1, net->dZ2, net->W2t);

    /* A1grad = A1 * (1 - A1) */
    mat_apply(net->A1grad, net->A1, sigmoid_derivative_from_output);
    mat_hadamard(net->dZ1, net->dA1, net->A1grad);

    mat_transpose(net->Xt, X);
    mat_mul(net->dW1, net->Xt, net->dZ1);
    mat_scale(net->dW1, net->dW1, inv_m);
    column_mean(net->db1, net->dZ1);
}

void net_update(Network *net, double learning_rate) {
    step_down(net->W1, net->dW1, learning_rate);
    step_down(net->b1, net->db1, learning_rate);
    step_down(net->W2, net->dW2, learning_rate);
    step_down(net->b2, net->db2, learning_rate);
}

double net_train_step(Network *net, const Matrix *X, const Matrix *Y,
                      double learning_rate) {
    net_forward(net, X);
    double loss = net_loss(net, Y);
    net_backward(net, X, Y);
    net_update(net, learning_rate);
    return loss;
}

/* ================= 기울기 검증 ================= */

/*
 * 역전파로 구한 기울기가 정말 맞는지 확인하는 방법.
 *
 * 가중치 하나를 아주 조금(h) 늘렸다 줄이면서 손실 변화를 직접 재면
 * 그 가중치의 기울기를 근사할 수 있다 (중앙 차분).
 *
 *   수치 기울기 ≈ ( L(w + h) - L(w - h) ) / (2h)
 *
 * 이 값과 역전파 결과가 거의 같으면 역전파 구현이 맞다.
 * 느려서 학습에는 쓸 수 없지만, 검증에는 아주 유용하다.
 */
static double check_one(Network *net, const Matrix *X, const Matrix *Y,
                        Matrix *param, const Matrix *grad, int index) {
    const double h = 1e-5;
    double original = param->data[index];

    param->data[index] = original + h;
    net_forward(net, X);
    double loss_plus = net_loss(net, Y);

    param->data[index] = original - h;
    net_forward(net, X);
    double loss_minus = net_loss(net, Y);

    param->data[index] = original;            /* 원래 값으로 복원 */

    double numeric  = (loss_plus - loss_minus) / (2.0 * h);
    double analytic = grad->data[index];

    /* 상대 오차: 값의 크기에 상관없이 비교하기 위해 */
    double denom = fabs(numeric) + fabs(analytic);
    if (denom < 1e-12) {
        return 0.0;
    }
    return fabs(numeric - analytic) / denom;
}

double net_gradient_check(Network *net, const Matrix *X, const Matrix *Y) {
    /* 먼저 역전파로 기울기를 구해 둔다 */
    net_forward(net, X);
    net_backward(net, X, Y);

    Matrix *params[] = { net->W1, net->b1, net->W2, net->b2 };
    Matrix *grads[]  = { net->dW1, net->db1, net->dW2, net->db2 };
    const char *names[] = { "W1", "b1", "W2", "b2" };

    double worst = 0.0;
    for (int p = 0; p < 4; p++) {
        int n = params[p]->rows * params[p]->cols;
        double worst_here = 0.0;
        for (int i = 0; i < n; i++) {
            double err = check_one(net, X, Y, params[p], grads[p], i);
            if (err > worst_here) {
                worst_here = err;
            }
        }
        printf("  %-3s 원소 %3d개, 최대 상대 오차 %.3e\n",
               names[p], n, worst_here);
        if (worst_here > worst) {
            worst = worst_here;
        }
    }
    return worst;
}
