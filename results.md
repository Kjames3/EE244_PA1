=================================================================
PART (a) — Baseline Decision Tree | Mushroom Dataset
=================================================================
[INFO] mushrooms.csv not found — generating synthetic stand-in (500 rows).

Dataset shape : (500, 8)
Target counts :
class
e    422
p     78

Baseline DT  — train error: 0.0000  |  test error: 0.1300
Tree depth   : 15   |  leaves: 55

Classification Report (test set):
              precision    recall  f1-score   support

           0       0.95      0.89      0.92        84
           1       0.57      0.75      0.65        16

    accuracy                           0.87       100
   macro avg       0.76      0.82      0.78       100
weighted avg       0.89      0.87      0.88       100

=================================================================
PART (b) — Pruning Experiments + k-NN Comparison
=================================================================
  max_depth=   1  |  train err=0.1550  |  test err=0.1600
  max_depth=   2  |  train err=0.0675  |  test err=0.0500
  max_depth=   3  |  train err=0.0675  |  test err=0.0500
  max_depth=   4  |  train err=0.0675  |  test err=0.0500
  max_depth=   5  |  train err=0.0600  |  test err=0.0700
  max_depth=   7  |  train err=0.0400  |  test err=0.1000
  max_depth=  10  |  train err=0.0175  |  test err=0.1300
  max_depth=  15  |  train err=0.0000  |  test err=0.1300
  max_depth=  20  |  train err=0.0000  |  test err=0.1300
  max_depth=None  |  train err=0.0000  |  test err=0.1300

k-NN sweep:
  k= 1  test err=0.2700
  k= 3  test err=0.2200
  k= 5  test err=0.1900
  k= 7  test err=0.2000
  k=11  test err=0.1800
  k=15  test err=0.1600
  k=21  test err=0.1600

Best k-NN: k=15, test error=0.1600

[Saved] part_b_pruning.png

=================================================================
PART (c) — Learning Curves | Mushroom Dataset
=================================================================
   10%  n=   35  time=0.978 ms  test_err=0.3000
   20%  n=   70  time=0.811 ms  test_err=0.1133
   30%  n=  105  time=0.831 ms  test_err=0.1933
   40%  n=  140  time=0.837 ms  test_err=0.1867
   50%  n=  175  time=0.859 ms  test_err=0.1733
   60%  n=  210  time=0.937 ms  test_err=0.1400
   70%  n=  244  time=0.887 ms  test_err=0.1533
[Saved] part_c_learning_curves.png

=================================================================
PART (d) — Random Forest | Loan Prediction Dataset
=================================================================

Dataset shape : (614, 13)
Target counts :
Loan_Status
Y    422
N    192

Best DT (max_depth=5)  — train err: 0.1752  |  test err: 0.1789
  RF n_trees= 10  |  train err=0.0163  |  test err=0.2114
  RF n_trees= 25  |  train err=0.0020  |  test err=0.1545
  RF n_trees= 50  |  train err=0.0000  |  test err=0.1789
  RF n_trees=100  |  train err=0.0000  |  test err=0.1707
  RF n_trees=150  |  train err=0.0000  |  test err=0.1545
  RF n_trees=200  |  train err=0.0000  |  test err=0.1545
  RF n_trees=300  |  train err=0.0000  |  test err=0.1463

Best RF: n_estimators=300, test error=0.1463
Improvement over best DT: 3.25 pp

Classification Report (RF, test set):
              precision    recall  f1-score   support

         0.0       0.86      0.63      0.73        38
         1.0       0.85      0.95      0.90        85

    accuracy                           0.85       123
   macro avg       0.85      0.79      0.81       123
weighted avg       0.85      0.85      0.85       123

[Saved] part_d_random_forest.png

=================================================================
SUMMARY — All Results
=================================================================
                   Model Train Error Test Error
      DT Baseline (Mush)      0.0000     0.1300
DT Pruned depth=5 (Loan)      0.1752     0.1789
   Best k-NN k=15 (Mush)           —     0.1600
    Best RF n=300 (Loan)      0.0000     0.1463

[Done] All figures saved.