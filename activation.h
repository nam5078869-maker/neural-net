#ifndef ACTIVATION_H
#define ACTIVATION_H

/* 시그모이드: 어떤 실수든 0~1 사이 값으로 바꾼다.
 * sigmoid(x) = 1 / (1 + e^(-x)) */
double sigmoid(double x);

/* 시그모이드의 미분값을 "출력값 a"로 계산한다.
 * a = sigmoid(x) 일 때 sigmoid'(x) = a * (1 - a)
 * (3단계 역전파에서 사용) */
double sigmoid_derivative_from_output(double a);

#endif
