/*
 * 6단계: 저장한 모델로 예측하기
 *
 * 학습을 다시 하지 않고 model.txt를 읽어 바로 숫자를 맞힌다.
 *
 * 실행: ./predict [이미지 번호 ...]
 *   ./predict           → 시험 데이터 전체 정확도와 혼동 행렬
 *   ./predict 0 1 2     → 해당 번호 이미지를 그리고 확률까지 보여준다
 */
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "mnist.h"
#include "model.h"
#include "network.h"

#define DATA_DIR   "data"
#define MODEL_PATH "model.txt"
#define BATCH      100
#define TEST_COUNT 10000

static int argmax_row(const Matrix *m, int row) {
    int best = 0;
    for (int j = 1; j < m->cols; j++) {
        if (MAT_AT(m, row, j) > MAT_AT(m, row, best)) {
            best = j;
        }
    }
    return best;
}

/* 이미지 한 장을 배치의 첫 줄에 넣고 예측한다 (나머지 줄은 쓰지 않는다) */
static void predict_one(Network *net, Matrix *X, const Dataset *ds, int index) {
    mat_fill(X, 0.0);
    for (int p = 0; p < MNIST_IMAGE_SIZE; p++) {
        MAT_AT(X, 0, p) = MAT_AT(ds->images, index, p);
    }
    net_forward(net, X);

    int guess = argmax_row(net->A2, 0);
    int answer = ds->digits[index];

    mnist_print_image(ds, index);
    printf("예측: %d  (정답 %d) %s\n", guess, answer,
           guess == answer ? "맞음" : "틀림");

    printf("확률:\n");
    for (int d = 0; d < MNIST_CLASSES; d++) {
        double p = MAT_AT(net->A2, 0, d);
        printf("  %d %6.2f%% ", d, p * 100.0);
        int bars = (int)(p * 40.0 + 0.5);
        for (int b = 0; b < bars; b++) {
            putchar('#');
        }
        putchar('\n');
    }
}

/* 시험 데이터 전체 정확도 + 혼동 행렬 */
static void evaluate_all(Network *net, Matrix *X, const Dataset *ds) {
    int confusion[MNIST_CLASSES][MNIST_CLASSES] = {{0}};
    int batches = ds->count / BATCH;
    int correct = 0;

    for (int b = 0; b < batches; b++) {
        for (int i = 0; i < BATCH; i++) {
            int src = b * BATCH + i;
            for (int p = 0; p < MNIST_IMAGE_SIZE; p++) {
                MAT_AT(X, i, p) = MAT_AT(ds->images, src, p);
            }
        }
        net_forward(net, X);

        for (int i = 0; i < BATCH; i++) {
            int answer = ds->digits[b * BATCH + i];
            int guess  = argmax_row(net->A2, i);
            confusion[answer][guess]++;
            if (answer == guess) {
                correct++;
            }
        }
    }

    printf("시험 데이터 %d장 중 %d장 정답 → 정확도 %.2f%%\n",
           batches * BATCH, correct, 100.0 * correct / (batches * BATCH));

    printf("\n[혼동 행렬] 세로: 정답, 가로: 예측\n");
    printf("      ");
    for (int j = 0; j < MNIST_CLASSES; j++) {
        printf("%5d", j);
    }
    printf("   정확도\n");

    for (int i = 0; i < MNIST_CLASSES; i++) {
        printf("  %d | ", i);
        int total = 0;
        for (int j = 0; j < MNIST_CLASSES; j++) {
            total += confusion[i][j];
        }
        for (int j = 0; j < MNIST_CLASSES; j++) {
            if (confusion[i][j] == 0) {
                printf("    .");
            } else {
                printf("%5d", confusion[i][j]);
            }
        }
        printf("   %5.1f%%\n", total ? 100.0 * confusion[i][i] / total : 0.0);
    }
    printf("\n대각선이 맞힌 개수다. 대각선에서 벗어난 큰 값이 자주 헷갈리는 조합이다.\n");
}

int main(int argc, char **argv) {
    Network *net = net_load(MODEL_PATH, BATCH);
    if (net == NULL) {
        fprintf(stderr, "\n먼저 학습해서 모델을 만들어 주세요:\n"
                        "  ./mnist_train 60000 20 128 100 0.5\n");
        return 1;
    }

    Dataset *test = mnist_load(DATA_DIR "/t10k-images-idx3-ubyte",
                               DATA_DIR "/t10k-labels-idx1-ubyte",
                               TEST_COUNT);
    if (test == NULL) {
        net_free(net);
        return 1;
    }

    Matrix *X = mat_create(BATCH, MNIST_IMAGE_SIZE);
    if (X == NULL) {
        mnist_free(test);
        net_free(net);
        return 1;
    }

    printf("모델을 불러왔습니다: %d -> %d -> %d\n\n",
           net->n_in, net->n_hidden, net->n_out);

    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            int index = atoi(argv[i]);
            if (index < 0 || index >= test->count) {
                printf("0~%d 사이의 번호를 입력해 주세요.\n", test->count - 1);
                continue;
            }
            predict_one(net, X, test, index);
            printf("\n");
        }
    } else {
        evaluate_all(net, X, test);
    }

    mat_free(X);
    mnist_free(test);
    net_free(net);
    return 0;
}
