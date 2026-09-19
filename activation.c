#include <math.h>
#include "activation.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative_from_output(double a) {
    return a * (1.0 - a);
}
