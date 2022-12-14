### Assignment: multilabel_classification_sgd
#### Date: Deadline: Nov 14, 7:59 a.m.
#### Points: 3 points
#### Examples: multilabel_classification_sgd_examples
#### Tests: multilabel_classification_sgd_tests

Starting with the [multilabel_classification_sgd.py](https://github.com/ufal/npfl129/tree/master/labs/05/multilabel_classification_sgd.py),
implement minibatch SGD for multi-label classification and
manually compute micro-averaged and macro-averaged $F_1$-score.

#### Examples Start: multilabel_classification_sgd_examples
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 multilabel_classification_sgd.py --batch_size=10 --epochs=9 --classes=5`
```
After epoch 1: train F1 micro 56.45% macro 46.71%, test F1 micro 58.25% macro 43.9%
After epoch 2: train F1 micro 71.46% macro 59.47%, test F1 micro 73.77% macro 60.3%
After epoch 3: train F1 micro 73.06% macro 61.02%, test F1 micro 71.71% macro 56.8%
After epoch 4: train F1 micro 77.30% macro 66.48%, test F1 micro 76.19% macro 64.1%
After epoch 5: train F1 micro 76.05% macro 67.34%, test F1 micro 74.46% macro 61.4%
After epoch 6: train F1 micro 78.22% macro 73.24%, test F1 micro 77.40% macro 66.1%
After epoch 7: train F1 micro 78.13% macro 73.33%, test F1 micro 74.41% macro 61.7%
After epoch 8: train F1 micro 78.92% macro 74.73%, test F1 micro 76.78% macro 66.9%
After epoch 9: train F1 micro 80.76% macro 76.31%, test F1 micro 78.18% macro 68.3%
Learned weights:
  -0.09 -0.17 -0.16 -0.01 0.09 0.01 0.04 -0.09 0.04 0.07 ...
  -0.08 0.09 0.02 -0.07 -0.08 -0.13 -0.07 0.09 0.06 0.01 ...
  0.20 0.25 0.09 0.00 0.02 -0.18 -0.18 -0.15 0.06 0.07 ...
  0.06 -0.04 -0.07 -0.01 0.10 0.13 0.10 0.17 0.20 -0.01 ...
  0.06 -0.11 -0.12 -0.05 -0.20 0.04 -0.01 -0.03 -0.16 -0.11 ...
```
- `python3 multilabel_classification_sgd.py --batch_size=10 --epochs=9 --classes=10 --data_size=20`
```
After epoch 1: train F1 micro 18.75% macro 10.08%, test F1 micro 8.70% macro 5.4%
After epoch 2: train F1 micro 21.43% macro 6.00%, test F1 micro 10.00% macro 5.4%
After epoch 3: train F1 micro 24.00% macro 6.00%, test F1 micro 5.71% macro 2.5%
After epoch 4: train F1 micro 25.00% macro 6.00%, test F1 micro 5.88% macro 2.5%
After epoch 5: train F1 micro 26.09% macro 6.00%, test F1 micro 6.06% macro 2.5%
After epoch 6: train F1 micro 26.09% macro 6.00%, test F1 micro 6.06% macro 2.5%
After epoch 7: train F1 micro 33.33% macro 7.27%, test F1 micro 6.06% macro 2.5%
After epoch 8: train F1 micro 33.33% macro 7.27%, test F1 micro 6.06% macro 2.5%
After epoch 9: train F1 micro 33.33% macro 7.27%, test F1 micro 6.06% macro 2.5%
Learned weights:
  -0.07 -0.08 0.03 0.02 -0.11 0.09 0.00 0.02 0.05 -0.11 ...
  0.16 0.07 -0.06 -0.05 0.06 0.03 -0.07 -0.06 0.05 0.09 ...
  -0.01 0.08 -0.08 -0.12 -0.09 0.07 0.03 -0.13 -0.07 0.02 ...
  -0.05 -0.09 -0.02 0.14 0.01 0.06 0.01 -0.01 -0.08 -0.03 ...
  -0.12 -0.09 -0.06 0.04 -0.10 -0.01 -0.08 -0.02 -0.08 0.01 ...
  -0.12 -0.09 0.00 0.01 -0.02 0.06 -0.03 -0.02 -0.08 -0.05 ...
  -0.15 -0.06 -0.11 -0.08 -0.09 -0.11 -0.11 -0.01 0.00 -0.05 ...
  0.04 -0.03 -0.06 -0.09 -0.04 -0.07 0.01 -0.12 -0.00 -0.04 ...
  -0.01 0.03 0.05 0.04 0.00 -0.10 -0.08 0.00 0.05 -0.10 ...
  0.04 -0.02 -0.09 -0.06 -0.10 -0.02 0.09 -0.09 -0.04 -0.10 ...
```
- `python3 multilabel_classification_sgd.py --batch_size=5 --epochs=9 --classes=5 --learning_rate=0.02`
```
After epoch 1: train F1 micro 60.66% macro 47.96%, test F1 micro 60.82% macro 46.6%
After epoch 2: train F1 micro 79.28% macro 77.99%, test F1 micro 77.65% macro 71.1%
After epoch 3: train F1 micro 80.27% macro 74.86%, test F1 micro 79.57% macro 69.6%
After epoch 4: train F1 micro 81.22% macro 79.85%, test F1 micro 77.41% macro 70.1%
After epoch 5: train F1 micro 80.50% macro 78.76%, test F1 micro 72.54% macro 65.1%
After epoch 6: train F1 micro 82.86% macro 81.46%, test F1 micro 75.62% macro 69.2%
After epoch 7: train F1 micro 81.19% macro 79.54%, test F1 micro 72.51% macro 65.3%
After epoch 8: train F1 micro 81.37% macro 79.59%, test F1 micro 75.06% macro 68.9%
After epoch 9: train F1 micro 83.83% macro 82.38%, test F1 micro 79.74% macro 74.3%
Learned weights:
  -0.18 -0.31 -0.23 0.05 0.12 -0.02 0.09 -0.25 0.21 0.16 ...
  -0.21 0.18 -0.12 -0.08 -0.13 -0.17 -0.12 0.15 0.10 0.04 ...
  0.47 0.32 0.13 0.01 0.09 -0.36 -0.29 -0.26 0.27 0.14 ...
  0.12 -0.07 -0.11 0.04 0.28 0.21 0.11 0.28 0.39 0.04 ...
  0.22 -0.24 -0.26 -0.03 -0.48 0.06 -0.10 0.01 -0.28 -0.14 ...
```
#### Examples End:
#### Tests Start: multilabel_classification_sgd_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 multilabel_classification_sgd.py --batch_size=10 --epochs=2 --classes=5`
```
After epoch 1: train F1 micro 56.45% macro 46.71%, test F1 micro 58.25% macro 43.9%
After epoch 2: train F1 micro 71.46% macro 59.47%, test F1 micro 73.77% macro 60.3%
Learned weights:
  -0.05 -0.11 -0.12 -0.05 0.04 0.04 0.02 0.01 -0.05 0.03 ...
  0.05 -0.01 0.09 -0.05 -0.06 -0.08 -0.05 0.02 0.03 0.00 ...
  0.10 0.16 0.08 0.01 -0.02 -0.05 -0.11 -0.09 -0.04 0.05 ...
  0.03 0.00 -0.06 -0.01 0.01 0.06 0.10 0.08 0.12 0.01 ...
  -0.03 -0.02 -0.08 -0.05 -0.07 -0.05 0.06 -0.03 -0.09 -0.09 ...
```
- `python3 multilabel_classification_sgd.py --batch_size=10 --epochs=2 --classes=10 --data_size=20`
```
After epoch 1: train F1 micro 18.75% macro 10.08%, test F1 micro 8.70% macro 5.4%
After epoch 2: train F1 micro 21.43% macro 6.00%, test F1 micro 10.00% macro 5.4%
Learned weights:
  -0.04 -0.09 0.02 0.02 -0.09 0.09 -0.02 0.04 0.06 -0.09 ...
  0.11 0.09 -0.07 -0.06 0.02 0.05 -0.05 -0.06 0.04 0.06 ...
  0.02 0.06 -0.06 -0.10 -0.10 0.08 0.05 -0.12 -0.05 0.04 ...
  -0.01 -0.07 -0.03 0.09 0.05 0.07 -0.03 0.03 -0.10 -0.01 ...
  -0.09 -0.08 -0.03 0.07 -0.07 0.01 -0.06 0.01 -0.06 0.03 ...
  -0.10 -0.08 0.03 0.04 0.01 0.07 -0.01 0.02 -0.06 -0.02 ...
  -0.12 -0.05 -0.08 -0.06 -0.06 -0.09 -0.09 0.03 0.02 -0.02 ...
  0.05 -0.01 -0.02 -0.09 -0.02 -0.07 0.04 -0.11 0.01 -0.03 ...
  0.01 -0.00 0.02 0.04 0.01 -0.09 -0.09 -0.02 0.07 -0.10 ...
  0.04 -0.04 -0.09 -0.02 -0.07 -0.03 0.09 -0.08 -0.01 -0.08 ...
```
- `python3 multilabel_classification_sgd.py --batch_size=5 --epochs=2 --classes=5 --learning_rate=0.02`
```
After epoch 1: train F1 micro 60.66% macro 47.96%, test F1 micro 60.82% macro 46.6%
After epoch 2: train F1 micro 79.28% macro 77.99%, test F1 micro 77.65% macro 71.1%
Learned weights:
  -0.08 -0.15 -0.14 -0.01 0.09 0.03 0.04 -0.08 0.03 0.08 ...
  -0.06 0.09 0.04 -0.06 -0.08 -0.13 -0.06 0.11 0.07 0.01 ...
  0.21 0.28 0.12 0.03 0.02 -0.16 -0.16 -0.14 0.06 0.13 ...
  0.07 -0.00 -0.04 0.00 0.12 0.13 0.11 0.19 0.21 0.03 ...
  0.07 -0.10 -0.10 -0.04 -0.19 0.05 0.01 -0.03 -0.15 -0.10 ...
```
#### Tests End:
