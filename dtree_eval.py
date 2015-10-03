import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score



def evaluatePerformance(numTrials=100):
    
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    none_accuracy_list = []
    stump_accuracy_list = []
    three_accuracy_list = []
    for i in range(numTrials):
        idx = np.arange(n)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
        X_folds = {}
        y_folds = {}
        for i,j in zip(range(0,163,27),range(1,8)):
            X_fold = X[i:(i+27),:]
            X_folds[j] = X_fold
            y_fold = y[i:(i+27),:]
            y_folds[j] = y_fold
        for i,j in zip(range(190,268,26),range(8,11)):
            X_fold = X[i:(i+27),:]
            X_folds[j] = X_fold
            y_fold = y[i:(i+27),:]
            y_folds[j] = y_fold
        
        dup_X_folds = X_folds
        dup_y_folds = y_folds
        X_train_dict = {}
        y_train_dict = {}
        for i in range(1,11):
            dup_X_folds.pop(i,None)
            dup_y_folds.pop(i,None)
            X_train_dict[i] = dup_X_folds
            y_train_dict[i] = dup_y_folds
            X_folds = {}
            y_folds = {}
            for i,j in zip(range(0,163,27),range(1,8)):
                X_fold = X[i:(i+27),:]
                X_folds[j] = X_fold
                y_fold = y[i:(i+27),:]
                y_folds[j] = y_fold
            for i,j in zip(range(190,268,26),range(8,11)):
                X_fold = X[i:(i+27),:]
                X_folds[j] = X_fold
                y_fold = y[i:(i+27),:]
                y_folds[j] = y_fold
            dup_X_folds = X_folds
            dup_y_folds = y_folds
    
        for i in range(1,11):
            if i == 1:
                X_stack1 = X_train_dict[i][2]
                y_stack1 = y_train_dict[i][2]
                for j in range(3,11):
                    X_stack1 = np.vstack([X_stack1,X_train_dict[i][j]])
                    y_stack1 = np.vstack([y_stack1,y_train_dict[i][j]])
            elif i == 2:
                X_stack2 = X_train_dict[i][1]
                y_stack2 = y_train_dict[i][1]
                for j in [3,4,5,6,7,8,9,10]:
                    X_stack2 = np.vstack([X_stack2,X_train_dict[i][j]])
                    y_stack2 = np.vstack([y_stack2,y_train_dict[i][j]])
            elif i == 3:
                X_stack3 = X_train_dict[i][1]
                y_stack3 = y_train_dict[i][1]
                for j in [2,4,5,6,7,8,9,10]:
                    X_stack3 = np.vstack([X_stack3,X_train_dict[i][j]])
                    y_stack3 = np.vstack([y_stack3,y_train_dict[i][j]])
            elif i == 4:
                X_stack4 = X_train_dict[i][1]
                y_stack4 = y_train_dict[i][1]
                for j in [2,3,5,6,7,8,9,10]:
                    X_stack4 = np.vstack([X_stack4,X_train_dict[i][j]])
                    y_stack4 = np.vstack([y_stack4,y_train_dict[i][j]])
            elif i == 5:
                X_stack5 = X_train_dict[i][1]
                y_stack5 = y_train_dict[i][1]
                for j in [2,3,4,6,7,8,9,10]:
                    X_stack5 = np.vstack([X_stack5,X_train_dict[i][j]])
                    y_stack5 = np.vstack([y_stack5,y_train_dict[i][j]])
            elif i == 6:
                X_stack6 = X_train_dict[i][1]
                y_stack6 = y_train_dict[i][1]
                for j in [2,3,4,5,7,8,9,10]:
                    X_stack6 = np.vstack([X_stack6,X_train_dict[i][j]])
                    y_stack6 = np.vstack([y_stack6,y_train_dict[i][j]])
            elif i == 7:
                X_stack7 = X_train_dict[i][1]
                y_stack7 = y_train_dict[i][1]
                for j in [2,3,4,5,6,8,9,10]:
                    X_stack7 = np.vstack([X_stack7,X_train_dict[i][j]])
                    y_stack7 = np.vstack([y_stack7,y_train_dict[i][j]])
            elif i == 8:
                X_stack8 = X_train_dict[i][1]
                y_stack8 = y_train_dict[i][1]
                for j in [2,3,4,5,6,7,9,10]:
                    X_stack8 = np.vstack([X_stack8,X_train_dict[i][j]])
                    y_stack8 = np.vstack([y_stack8,y_train_dict[i][j]])
            elif i == 9:
                X_stack9 = X_train_dict[i][1]
                y_stack9 = y_train_dict[i][1]
                for j in [2,3,4,5,6,7,8,10]:
                    X_stack9 = np.vstack([X_stack9,X_train_dict[i][j]])
                    y_stack9 = np.vstack([y_stack9,y_train_dict[i][j]])
            elif i == 10:
                X_stack10 = X_train_dict[i][1]
                y_stack10 = y_train_dict[i][1]
                for j in [2,3,4,5,6,7,8,9]:
                    X_stack10 = np.vstack([X_stack10,X_train_dict[i][j]])
                    y_stack10 = np.vstack([y_stack10,y_train_dict[i][j]])
        X_stacks = [X_stack1,X_stack2,X_stack3,X_stack4,X_stack5,
                    X_stack6,X_stack7,X_stack8,X_stack9,X_stack10]
        y_stacks = [y_stack1,y_stack2,y_stack3,y_stack4,y_stack5,
                    y_stack6,y_stack7,y_stack8,y_stack9,y_stack10]
        
        for t in [None,1,3]:
            clf = tree.DecisionTreeClassifier(max_depth=t)
            for i,j,k in zip(X_stacks,y_stacks,range(1,11)):    
                clf = clf.fit(i,j)
                y_pred = clf.predict(X_folds[k])
                accuracy = accuracy_score(y_folds[k], y_pred)
            if t == None:                
                none_accuracy_list.append(accuracy)
            if t == 1:
                stump_accuracy_list.append(accuracy)
            if t == 3:
                three_accuracy_list.append(accuracy)
                
    meanDecisionTreeAccuracy = np.mean(none_accuracy_list)    
    stddevDecisionTreeAccuracy = np.std(none_accuracy_list)
    meanDecisionStumpAccuracy = np.mean(stump_accuracy_list)
    stddevDecisionStumpAccuracy = np.std(stump_accuracy_list)
    meanDT3Accuracy = np.mean(three_accuracy_list)
    stddevDT3Accuracy = np.std(three_accuracy_list)

    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats

if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print "Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")"
    print "Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")"
    print "3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")"