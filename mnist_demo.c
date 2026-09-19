/*
 * 4단계: MNIST 데이터가 제대로 읽히는지 확인하기
 *
 * 학습은 아직 하지 않는다. 데이터를 잘못 읽으면 5단계에서
 * "왜 학습이 안 되지?" 하고 한참 헤매게 되므로 여기서 확인해 둔다.
 */
#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"

#define DATA_DIR "data"
#define DEMO_COUNT 2000      /* 확인용으로 일부만 읽는다 */

/* 라벨 0~9가 각각 몇 개인지 센다 */
static void print_label_distribution(const Dataset *ds) {
    int counts[MNIST_CLASSES] = {0};
    for (int i = 0; i < ds->count; i++) {
        counts[ds->digits[i]]++;
    }

    printf("\n[숫자별 개수]\n");
    for (int d = 0; d < MNIST_CLASSES; d++) {
        printf("  %d: %4d  ", d, counts[d]);
        int bars = counts[d] * 40 / ds->count;   /* 간단한 막대그래프 */
        for (int b = 0; b < bars; b++) {
            putchar('#');
        }
        putchar('\n');
    }
}

/* 픽셀 값이 0.0~1.0 범위에 있고, 원-핫 라벨의 합이 1인지 확인 */
static void sanity_check(const Dataset *ds) {
    double min = 1e9, max = -1e9, sum = 0.0;
    int n = ds->count * MNIST_IMAGE_SIZE;
    for (int i = 0; i < n; i++) {
        double v = ds->images->data[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
    }

    int onehot_ok = 1;
    for (int i = 0; i < ds->count; i++) {
        double row_sum = 0.0;
        for (int c = 0; c < MNIST_CLASSES; c++) {
            row_sum += MAT_AT(ds->labels, i, c);
        }
        if (row_sum != 1.0 || MAT_AT(ds->labels, i, ds->digits[i]) != 1.0) {
            onehot_ok = 0;
            break;
        }
    }

    printf("\n[검사]\n");
    printf("  픽셀 값 범위: %.3f ~ %.3f  %s\n", min, max,
           (min >= 0.0 && max <= 1.0) ? "정상" : "이상");
    printf("  픽셀 평균: %.3f (보통 0.13 근처)\n", sum / n);
    printf("  원-핫 라벨: %s\n", onehot_ok ? "정상" : "이상");
}

int main(void) {
    printf("MNIST 데이터를 읽는 중... (%s 폴더)\n", DATA_DIR);

    Dataset *train = mnist_load(DATA_DIR "/train-images-idx3-ubyte",
                                DATA_DIR "/train-labels-idx1-ubyte",
                                DEMO_COUNT);
    if (train == NULL) {
        fprintf(stderr,
                "\n데이터 파일을 찾을 수 없습니다.\n"
                "%s 폴더에 아래 4개 파일이 있어야 합니다:\n"
                "  train-images-idx3-ubyte\n"
                "  train-labels-idx1-ubyte\n"
                "  t10k-images-idx3-ubyte\n"
                "  t10k-labels-idx1-ubyte\n", DATA_DIR);
        return 1;
    }

    Dataset *test = mnist_load(DATA_DIR "/t10k-images-idx3-ubyte",
                               DATA_DIR "/t10k-labels-idx1-ubyte",
                               DEMO_COUNT);
    if (test == NULL) {
        mnist_free(train);
        return 1;
    }

    printf("학습용 %d장, 시험용 %d장 (각 %dx%d = 픽셀 %d개)\n",
           train->count, test->count, train->rows, train->cols,
           MNIST_IMAGE_SIZE);

    /* 앞에서 3장을 그려 본다 */
    for (int i = 0; i < 3; i++) {
        printf("\n");
        mnist_print_image(train, i);
    }

    print_label_distribution(train);
    sanity_check(train);

    printf("\n첫 이미지의 원-핫 라벨: [");
    for (int c = 0; c < MNIST_CLASSES; c++) {
        printf("%d%s", (int)MAT_AT(train->labels, 0, c),
               c < MNIST_CLASSES - 1 ? " " : "");
    }
    printf("]  (정답 %d)\n", train->digits[0]);

    mnist_free(train);
    mnist_free(test);
    return 0;
}
