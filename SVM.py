import preprocess
X_train, y_train, X_test, y_test =  preprocess.scaled_data()

from sklearn import svm, datasets
import timeit

# Create the SVC model object
C = 1.0 # SVM regularization parameter

from sklearn.metrics import confusion_matrix

svc1 = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr')

print "SVM1 Train: "
start_time = timeit.default_timer()
svc1.fit(X_train, y_train)
elapsed = timeit.default_timer() - start_time
print str(elapsed)

print "SVM1 Test: "
start_time = timeit.default_timer()
y_pred = svc1.predict(X_test)
elapsed = timeit.default_timer() - start_time
print str(elapsed)

score_rbf = svc1.score(X_test, y_test)

cm_rbf  = confusion_matrix(y_test, y_pred)