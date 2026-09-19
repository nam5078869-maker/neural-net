/*
 * 모델 파일 형식
 *
 *   NNMODEL 1                 <- 식별자와 형식 버전
 *   784 128 10 1              <- 입력 수, 은닉 수, 출력 수, 출력함수(0=sigmoid,1=softmax)
 *   W1                        <- 이름표 (사람이 읽기 좋게)
 *   0.0123 -0.0456 ...        <- 행마다 한 줄
 *   b1
 *   ...
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

#define MODEL_MAGIC   "NNMODEL"
#define MODEL_VERSION 1

/* 행렬 하나를 이름표와 함께 저장한다 */
static void write_matrix(FILE *fp, const char *name, const Matrix *m) {
    fprintf(fp, "%s\n", name);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            /* %.17g: double을 다시 읽었을 때 완전히 같은 값이 되는 자리수 */
            fprintf(fp, "%.17g%s", MAT_AT(m, i, j),
                    j == m->cols - 1 ? "\n" : " ");
        }
    }
}

/* 이름표를 확인하고 행렬을 읽는다. 성공 1, 실패 0 */
static int read_matrix(FILE *fp, const char *name, Matrix *m) {
    char label[32];
    if (fscanf(fp, "%31s", label) != 1 || strcmp(label, name) != 0) {
        fprintf(stderr, "모델 파일에서 %s를 찾을 수 없습니다.\n", name);
        return 0;
    }
    for (int i = 0; i < m->rows * m->cols; i++) {
        if (fscanf(fp, "%lf", &m->data[i]) != 1) {
            fprintf(stderr, "%s의 값이 부족합니다.\n", name);
            return 0;
        }
    }
    return 1;
}

int net_save(const Network *net, const char *path) {
    /* 6단계 축구 프로젝트와 같은 방식: 임시 파일에 쓰고 이름을 바꾼다 */
    char tmp_path[512];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

    FILE *fp = fopen(tmp_path, "w");
    if (fp == NULL) {
        return -1;
    }

    fprintf(fp, "%s %d\n", MODEL_MAGIC, MODEL_VERSION);
    fprintf(fp, "%d %d %d %d\n", net->n_in, net->n_hidden, net->n_out,
            net->output == OUT_SOFTMAX ? 1 : 0);

    write_matrix(fp, "W1", net->W1);
    write_matrix(fp, "b1", net->b1);
    write_matrix(fp, "W2", net->W2);
    write_matrix(fp, "b2", net->b2);

    int write_error = ferror(fp);
    if (fclose(fp) != 0 || write_error) {
        remove(tmp_path);
        return -1;
    }
    if (rename(tmp_path, path) != 0) {
        remove(tmp_path);
        return -1;
    }
    return 0;
}

Network *net_load(const char *path, int batch) {
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "모델 파일을 열 수 없습니다: %s\n", path);
        return NULL;
    }

    char magic[32];
    int version, n_in, n_hidden, n_out, softmax;
    if (fscanf(fp, "%31s %d", magic, &version) != 2
        || strcmp(magic, MODEL_MAGIC) != 0 || version != MODEL_VERSION) {
        fprintf(stderr, "모델 파일 형식이 아닙니다: %s\n", path);
        fclose(fp);
        return NULL;
    }
    if (fscanf(fp, "%d %d %d %d", &n_in, &n_hidden, &n_out, &softmax) != 4
        || n_in <= 0 || n_hidden <= 0 || n_out <= 0) {
        fprintf(stderr, "모델 구조 정보가 잘못되었습니다.\n");
        fclose(fp);
        return NULL;
    }

    Network *net = net_create(n_in, n_hidden, n_out, batch);
    if (net == NULL) {
        fprintf(stderr, "메모리가 부족합니다.\n");
        fclose(fp);
        return NULL;
    }
    net_use_softmax(net, softmax);

    if (!read_matrix(fp, "W1", net->W1) || !read_matrix(fp, "b1", net->b1)
        || !read_matrix(fp, "W2", net->W2) || !read_matrix(fp, "b2", net->b2)) {
        net_free(net);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return net;
}
