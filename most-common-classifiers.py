# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
def gaussian_classifier(X_train, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix    
    cm = confusion_matrix(y_test, y_pred)
        
    return cm

# Fitting classifier to random forest classifier entropy
from sklearn.ensemble import RandomForestClassifier
def random_forest_classifier(X_train, y_train):
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix    
    cm = confusion_matrix(y_test, y_pred)
        
    return cm


# Fitting classifier to knn
from sklearn.neighbors import KNeighborsClassifier
def knn_classifier(X_train, y_train):
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix    
    cm = confusion_matrix(y_test, y_pred)
    
    return cm

# Fitting classifier to SVC
from sklearn.svm import SVC
def svc_classifier(X_train, y_train):
    classifier = SVC(probability=False, kernel = 'rbf', gamma=.0080)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return cm

# CART - Fitting classifier to Decision Tree
from sklearn import tree
def decision_tree_classifier(X_train, y_train):
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return cm

# Maximum Entropy
from nltk.classify import MaxentClassifier
def maximum_entropy_classifier():
    classifier = MaxentClassifier(X_train, )
    

cm_naive_bayes = gaussian_classifier(X_train, y_train)
cm_random_forest = random_forest_classifier(X_train, y_train)
cm_knn = knn_classifier(X_train, y_train)    
cm_svc = svc_classifier(X_train, y_train)    
cm_decision_tree = decision_tree_classifier(X_train, y_train)




### ------- Evaluation
# TN = CM[0][0]
# FN = CM[1][0]
# TP = CM[1][1]
# FP = CM[0][1]

# Thus in binary classification, 
# the count of true negatives is C_{0,0}, 
# false negatives is C_{1,0}, 
# true positives is C_{1,1} and 
# false positives is C_{0,1}.
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html


def get_TN(cm):
    return cm[0][0]

def get_FN(cm):
    return cm[1][0]

def get_TP(cm):
    return cm[1][1]

def get_FP(cm):
    return cm[0][1]

# Calc accuracy 
# (TP + TN) / (TP + TN + FP + FN)
def calc_accuracy(cm):    
    accuracy = (get_TP(cm)+ get_TN(cm) ) / (get_TP(cm) + get_TN(cm) + get_FN(cm)  + get_FN(cm) )    
    return accuracy

# Calc precision
# TP / (TP + FP)
def calc_precision(cm):
    precision = get_TP(cm) / (get_TP(cm) + get_FN(cm) )
    return precision

# Calc recall
# TP / (TP + FN)
def calc_recall(cm):
    recall = get_TP(cm) / (get_TP(cm) + get_FN(cm) )
    return recall

# Calc F1 Score
# = 2 * Precision * Recall / (Precision + Recall)
def calc_f1_score(cm):
    f1_score = (2 * calc_precision(cm) * calc_recall(cm)) / (calc_precision(cm) + calc_recall(cm))
    return f1_score


acc_nb = calc_accuracy(cm_naive_bayes)
acc_rf = calc_accuracy(cm_random_forest)
acc_knn = calc_accuracy(cm_knn)
acc_svc = calc_accuracy(cm_svc)
acc_dt = calc_accuracy(cm_decision_tree)

prc_nb = calc_precision(cm_naive_bayes)
prc_rf = calc_precision(cm_random_forest)
prc_knn = calc_precision(cm_knn)
prc_svc = calc_precision(cm_svc)
prc_dt = calc_precision(cm_decision_tree)


rcl_nb = calc_recall(cm_naive_bayes)
rcl_rf = calc_recall(cm_random_forest)
rcl_knn = calc_recall(cm_knn)
rcl_svc = calc_recall(cm_svc)
rcl_dt = calc_recall(cm_decision_tree)

 
f1_score_nb = calc_f1_score(cm_naive_bayes)
f1_score_rf = calc_f1_score(cm_random_forest)
f1_score_knn = calc_f1_score(cm_knn)
f1_score_svc = calc_f1_score(cm_svc)
f1_score_dt = calc_f1_score(cm_decision_tree)



### ----- PRINT RESULTS

#float format
def ff(f):
    return "%0.2f" % f


def print_results_table():
    print("\t\tTP \t TN \t FP \t FN \t Acc \t Prc \t Rcl \t F1s")
    print("\nNaive Bayes:\t",
          get_TP(cm_naive_bayes),"\t",
          get_TN(cm_naive_bayes),"\t", 
          get_FP(cm_naive_bayes),"\t",
          get_FN(cm_naive_bayes),"\t",
          ff(acc_nb),"\t",ff(prc_nb),"\t",ff(rcl_nb),"\t",ff(f1_score_nb),
          "\nRandom Forest:\t",
          get_TP(cm_random_forest),"\t",
          get_TN(cm_random_forest),"\t", 
          get_FP(cm_random_forest),"\t",
          get_FN(cm_random_forest),"\t",
          ff(acc_rf),"\t",ff(prc_rf),"\t",ff(rcl_rf),"\t",ff(f1_score_rf),
          "\nK-Nearest N.:\t",
          get_TP(cm_knn),"\t",
          get_TN(cm_knn),"\t", 
          get_FP(cm_knn),"\t",
          get_FN(cm_knn),"\t",
          ff(acc_knn),"\t",ff(prc_knn),"\t",ff(rcl_knn),"\t",ff(f1_score_knn),
          "\nSupport V.M.:\t",
          get_TP(cm_svc),"\t",
          get_TN(cm_svc),"\t", 
          get_FP(cm_svc),"\t",
          get_FN(cm_svc),"\t",
          ff(acc_svc),"\t",ff(prc_svc),"\t",ff(rcl_svc),"\t",ff(f1_score_svc),
           "\nDecision Tree:\t",
          get_TP(cm_decision_tree),"\t",
          get_TN(cm_decision_tree),"\t", 
          get_FP(cm_decision_tree),"\t",
          get_FN(cm_decision_tree),"\t",
          ff(acc_dt),"\t",ff(prc_dt),"\t",ff(rcl_dt),"\t",ff(f1_score_dt)
          )
    
    
print_results_table()


def print_results():
    print("NBa: "," Acc: ",ff(acc_nb)," Prc: ",ff(prc_nb)," Rcl: ",ff(rcl_nb)," f1_score: ", ff(f1_score_nb))
    print("RFr: "," Acc: ",ff(acc_rf)," Prc: ",ff(prc_rf)," Rcl: ",ff(rcl_rf)," f1_score: ", ff(f1_score_rf))
    print("KNn: "," Acc: ",ff(acc_knn)," Prc: ",ff(prc_knn)," Rcl: ",ff(rcl_knn)," f1_score: ", ff(f1_score_knn))
    print("SVc: "," Acc: ",ff(acc_svc)," Prc: ",ff(prc_svc)," Rcl: ",ff(rcl_svc)," f1_score: ", ff(f1_score_svc))
    print("DEt: "," Acc: ",ff(acc_dt)," Prc: ",ff(prc_dt)," Rcl: ",ff(rcl_dt)," f1_score: ", ff(f1_score_dt))
