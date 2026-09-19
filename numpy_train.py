"""
7단계: 같은 신경망을 NumPy로 구현해 C 버전과 속도를 비교한다.

구조와 하이퍼파라미터를 mnist_train.c와 똑같이 맞췄다.
  784 -> 128 (sigmoid) -> 10 (softmax), 배치 100, 학습률 0.5

실행: python3 numpy_train.py [학습 장수] [에폭]
"""
import struct
import sys
import time

import numpy as np

DATA_DIR = "data"


def load_idx_images(path, count=None):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        if count:
            n = min(n, count)
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float64) / 255.0


def load_idx_labels(path, count=None):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        if count:
            n = min(n, count)
        digits = np.frombuffer(f.read(n), dtype=np.uint8)
    onehot = np.zeros((len(digits), 10))
    onehot[np.arange(len(digits)), digits] = 1.0
    return onehot, digits


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    train_count = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    hidden, batch, lr = 128, 100, 0.5

    rng = np.random.default_rng(42)

    X = load_idx_images(f"{DATA_DIR}/train-images-idx3-ubyte", train_count)
    Y, _ = load_idx_labels(f"{DATA_DIR}/train-labels-idx1-ubyte", train_count)
    Xt = load_idx_images(f"{DATA_DIR}/t10k-images-idx3-ubyte")
    _, yt = load_idx_labels(f"{DATA_DIR}/t10k-labels-idx1-ubyte")

    # C 버전과 같은 Xavier 초기화
    limit1 = np.sqrt(6.0 / (784 + hidden))
    limit2 = np.sqrt(6.0 / (hidden + 10))
    W1 = rng.uniform(-limit1, limit1, (784, hidden))
    b1 = np.zeros((1, hidden))
    W2 = rng.uniform(-limit2, limit2, (hidden, 10))
    b2 = np.zeros((1, 10))

    n = len(X)
    batches = n // batch
    print(f"NumPy {np.__version__} / 학습 {n}장, 배치 {batch}, {batches}회 갱신")
    print("에폭   손실      시험 정확도  걸린 시간")

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        order = rng.permutation(n)
        loss_sum = 0.0

        for b in range(batches):
            idx = order[b * batch:(b + 1) * batch]
            xb, yb = X[idx], Y[idx]

            A1 = sigmoid(xb @ W1 + b1)          # 순전파
            A2 = softmax(A1 @ W2 + b2)
            loss_sum += -np.sum(yb * np.log(A2 + 1e-12)) / batch

            dZ2 = A2 - yb                        # 역전파
            dW2 = A1.T @ dZ2 / batch
            db2 = dZ2.mean(axis=0, keepdims=True)
            dZ1 = (dZ2 @ W2.T) * A1 * (1 - A1)
            dW1 = xb.T @ dZ1 / batch
            db1 = dZ1.mean(axis=0, keepdims=True)

            W2 -= lr * dW2                       # 갱신
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        pred = softmax(sigmoid(Xt @ W1 + b1) @ W2 + b2).argmax(axis=1)
        acc = 100.0 * (pred == yt).mean()
        print(f"{epoch:4d}  {loss_sum / batches:8.5f}  {acc:9.2f}%  "
              f"{time.perf_counter() - start:7.1f}초")


if __name__ == "__main__":
    main()
