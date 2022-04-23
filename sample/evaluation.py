from sklearn import metrics
import numpy as np
import pandas as pd


# 模型训练、预测和性能评估
def get_metrics(true_labels, predicted_labels):
    print('Accuracy:', np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels), 4))
    print('Precision:', np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'), 4))
    print('Recall:', np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'), 4))
    print('F1 Score:', np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'), 4))


def train_predict_model(classifier,
                        train_features, train_labels,
                        test_features, test_labels):
    # 创建模型
    classifier.fit(train_features, train_labels)
    # 模型预测
    predictions = classifier.predict(test_features)
    return predictions


def display_confusion_matrix(true_labels, predicted_labels, classes=[1, 0]):
    total_classes = len(classes)
    level_labels = [total_classes * [0], list(range(total_classes))]

    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels,
                                  labels=classes)
    cm_frame = pd.DataFrame(data=cm,
                            columns=pd.MultiIndex(levels=[['Predicted:'], classes],
                                                  codes=level_labels),
                            index=pd.MultiIndex(levels=[['Actual:'], classes],
                                                codes=level_labels))
    print(cm_frame)


def display_classification_report(true_labels, predicted_labels, classes=[1, 0]):
    report = metrics.classification_report(y_true=true_labels,
                                           y_pred=predicted_labels,
                                           labels=classes)
    print(report)


def display_model_performance_metrics(true_labels, predicted_labels, classes=[1, 0]):
    print('Model Performance metrics:')
    print('-' * 30)
    get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    print('\nModel Classification report:')
    print('-' * 30)
    display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels,
                                  classes=classes)
    print('\nPrediction Confusion Matrix:')
    print('-' * 30)
    display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels,
                             classes=classes)
