/*
 * 5단계: MNIST 손글씨 숫자 분류 학습
 *
 *   입력 784 (28x28 픽셀)
 *     → 은닉층 128 (sigmoid)
 *     → 출력층 10 (softmax: 숫자 0~9일 확률)
 *
 * 미니배치 경사하강법으로 학습한다.
 *
 * 실행: ./mnist_train [학습 장수] [에폭] [은닉 뉴런] [배치] [학습률]
 *   예) ./mnist_train 60000 20 128 100 0.5
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "matrix.h"
#include "mnist.h"
#include "model.h"
#include "network.h"

#define DATA_DIR "data"
#define MODEL_PATH "model.txt"

/* 기본 설정 (명령줄 인자로 바꿀 수 있다) */
#define DEF_TRAIN_COUNT 20000
#define DEF_EPOCHS      10
#define DEF_HIDDEN      128
#define DEF_BATCH       100
#define DEF_LR          0.5
#define TEST_COUNT      10000
#define RANDOM_SEED     42

/* 한 행에서 가장 큰 값의 열 번호 = 신경망이 고른 숫자 */
static int argmax_row(const Matrix *m, int row) {
    int best = 0;
    for (int j = 1; j < m->cols; j++) {
        if (MAT_AT(m, row, j) > MAT_AT(m, row, best)) {
            best = j;
        }
    }
    return best;
}

/* 배열을 무작위로 섞는다 (피셔-예이츠).
 * 매 에폭 순서를 바꿔야 같은 순서에 치우쳐 학습하지 않는다. */
static void shuffle(int *array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

/* 데이터셋에서 골라낸 행들을 배치 행렬로 복사한다 */
static void fill_batch(Matrix *X, Matrix *Y, const Dataset *ds,
                       const int *indices, int start) {
    for (int i = 0; i < X->rows; i++) {
        int src = indices[start + i];
        memcpy(&MAT_AT(X, i, 0), &MAT_AT(ds->images, src, 0),
               sizeof(double) * MNIST_IMAGE_SIZE);
        memcpy(&MAT_AT(Y, i, 0), &MAT_AT(ds->labels, src, 0),
               sizeof(double) * MNIST_CLASSES);
    }
}

/*
 * 데이터셋 전체의 정확도를 잰다.
 * 신경망은 batch개씩만 계산할 수 있으므로 배치 단위로 나눠 처리한다.
 * wrong_index가 NULL이 아니면 틀린 예시를 최대 wrong_max개 기록한다.
 */
static double evaluate(Network *net, const Dataset *ds,
                       Matrix *X, Matrix *Y, const int *indices,
                       int max_samples,
                       int *wrong_index, int *wrong_guess, int wrong_max,
                       int *wrong_found) {
    int count = ds->count;
    if (max_samples > 0 && max_samples < count) {
        count = max_samples;      /* 학습 정확도는 일부만 재도 충분하다 */
    }
    int batches = count / net->batch;
    int correct = 0;
    int found = 0;

    for (int b = 0; b < batches; b++) {
        int start = b * net->batch;
        fill_batch(X, Y, ds, indices, start);
        net_forward(net, X);

        for (int i = 0; i < net->batch; i++) {
            int guess = argmax_row(net->A2, i);
            int answer = ds->digits[indices[start + i]];
            if (guess == answer) {
                correct++;
            } else if (wrong_index != NULL && found < wrong_max) {
                wrong_index[found] = indices[start + i];
                wrong_guess[found] = guess;
                found++;
            }
        }
    }

    if (wrong_found != NULL) {
        *wrong_found = found;
    }
    return 100.0 * correct / (batches * net->batch);
}

int main(int argc, char **argv) {
    int train_count = argc > 1 ? atoi(argv[1]) : DEF_TRAIN_COUNT;
    int epochs      = argc > 2 ? atoi(argv[2]) : DEF_EPOCHS;
    int hidden      = argc > 3 ? atoi(argv[3]) : DEF_HIDDEN;
    int batch       = argc > 4 ? atoi(argv[4]) : DEF_BATCH;
    double lr       = argc > 5 ? atof(argv[5]) : DEF_LR;

    if (train_count < batch || epochs < 1 || hidden < 1 || batch < 1
        || lr <= 0.0) {
        fprintf(stderr, "사용법: %s [학습 장수] [에폭] [은닉 뉴런] [배치] [학습률]\n",
                argv[0]);
        return 1;
    }

    srand(RANDOM_SEED);

    printf("MNIST 데이터를 읽는 중...\n");
    Dataset *train = mnist_load(DATA_DIR "/train-images-idx3-ubyte",
                                DATA_DIR "/train-labels-idx1-ubyte",
                                train_count);
    Dataset *test  = mnist_load(DATA_DIR "/t10k-images-idx3-ubyte",
                                DATA_DIR "/t10k-labels-idx1-ubyte",
                                TEST_COUNT);
    if (train == NULL || test == NULL) {
        mnist_free(train);
        mnist_free(test);
        return 1;
    }

    Network *net = net_create(MNIST_IMAGE_SIZE, hidden, MNIST_CLASSES, batch);
    Matrix *X = mat_create(batch, MNIST_IMAGE_SIZE);
    Matrix *Y = mat_create(batch, MNIST_CLASSES);
    int *train_order = malloc(sizeof(int) * train->count);
    int *test_order  = malloc(sizeof(int) * test->count);

    if (net == NULL || X == NULL || Y == NULL
        || train_order == NULL || test_order == NULL) {
        fprintf(stderr, "메모리가 부족합니다.\n");
        return 1;
    }
    net_use_softmax(net, 1);

    for (int i = 0; i < train->count; i++) train_order[i] = i;
    for (int i = 0; i < test->count; i++)  test_order[i] = i;

    int batches = train->count / batch;
    printf("\n구조: %d -> %d(sigmoid) -> %d(softmax)\n",
           MNIST_IMAGE_SIZE, hidden, MNIST_CLASSES);
    printf("학습 %d장, 시험 %d장, 배치 %d (에폭당 %d회 갱신), 학습률 %.3f\n",
           train->count, test->count, batch, batches, lr);
    printf("가중치 개수: %d\n\n",
           MNIST_IMAGE_SIZE * hidden + hidden + hidden * MNIST_CLASSES
           + MNIST_CLASSES);

    printf("에폭   손실      학습 정확도  시험 정확도  학습 시간  평가 시간\n");
    printf("----  --------  -----------  -----------  ---------  ---------\n");

    double best = 0.0;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        clock_t start_time = clock();

        shuffle(train_order, train->count);

        double loss_sum = 0.0;
        for (int b = 0; b < batches; b++) {
            fill_batch(X, Y, train, train_order, b * batch);
            loss_sum += net_train_step(net, X, Y, lr);
        }

        /* 학습에 걸린 시간과 정확도 측정에 걸린 시간을 따로 잰다 */
        clock_t eval_time = clock();
        double train_seconds = (double)(eval_time - start_time) / CLOCKS_PER_SEC;

        double train_acc = evaluate(net, train, X, Y, train_order,
                                    10000, NULL, NULL, 0, NULL);
        double test_acc  = evaluate(net, test, X, Y, test_order,
                                    0, NULL, NULL, 0, NULL);
        double eval_seconds = (double)(clock() - eval_time) / CLOCKS_PER_SEC;

        if (test_acc > best) {
            best = test_acc;
        }
        printf("%4d  %8.5f  %9.2f%%  %9.2f%%  %7.2f초  %7.2f초\n",
               epoch, loss_sum / batches, train_acc, test_acc,
               train_seconds, eval_seconds);
        fflush(stdout);
    }

    printf("\n가장 높은 시험 정확도: %.2f%%\n", best);

    /* 학습한 가중치를 저장한다. ./predict로 바로 쓸 수 있다 */
    if (net_save(net, MODEL_PATH) == 0) {
        printf("모델을 %s에 저장했습니다.\n", MODEL_PATH);
    } else {
        fprintf(stderr, "모델 저장에 실패했습니다.\n");
    }

    /* ---- 틀린 예시 보기 ---- */
    int wrong_index[3], wrong_guess[3], wrong_found = 0;
    evaluate(net, test, X, Y, test_order, 0,
             wrong_index, wrong_guess, 3, &wrong_found);

    printf("\n[틀린 예시 %d개]\n", wrong_found);
    for (int i = 0; i < wrong_found; i++) {
        printf("\n신경망의 답: %d\n", wrong_guess[i]);
        mnist_print_image(test, wrong_index[i]);
    }

    free(train_order);
    free(test_order);
    mat_free(X);
    mat_free(Y);
    net_free(net);
    mnist_free(train);
    mnist_free(test);
    return 0;
}
