#ifndef MODEL_H
#define MODEL_H

#include "network.h"

/*
 * 학습한 가중치를 텍스트 파일로 저장하고 불러온다.
 *
 * 텍스트로 저장하는 이유:
 *  - 메모장으로 열어 값을 확인할 수 있다
 *  - 4단계 MNIST 파일처럼 엔디안을 신경 쓸 필요가 없다
 *  - %.17g로 쓰면 double 값이 손실 없이 그대로 복원된다
 */

/* 가중치를 파일에 저장한다. 성공: 0 / 실패: -1 */
int net_save(const Network *net, const char *path);

/*
 * 파일에서 신경망을 읽어 새로 만든다 (구조는 파일에 적힌 값을 따른다).
 * batch는 한 번에 처리할 샘플 수로, 저장할 때와 달라도 된다.
 * 실패하면 NULL을 돌려주고 이유를 stderr에 출력한다.
 */
Network *net_load(const char *path, int batch);

#endif
