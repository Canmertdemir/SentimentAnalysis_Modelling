import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

from Text_Preprocess.text_preprocess_tools import text_preprocessing, train_test_sep, sentiment_analysis, world_to_vec

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

data_set = pd.read_excel(r'C:\Users\pc\SentimentAnalysis\Data_Set\amazon.xlsx')


text_preprocessing(data_set)
sentiment_analysis(data_set)
data_set, X_train, X_test, y_train, y_test = train_test_sep(data_set)
X_train_tf_idf_word, X_test_tf_idf_word = world_to_vec(X_train, X_test)
def feature_importance(df, Xtrain, ytrain, save_path=None):
    lgbm = LGBMClassifier()
    lgbm.fit(Xtrain, ytrain)
    feature_names = Xtrain.columns
    important_features = pd.Series(lgbm.feature_importances_, index=feature_names)

    fig, ax = plt.subplots()
    important_features.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Feature Importance")
    ax.set_xlabel("Features")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def model_stacking(Xtrain, Xtest, ytrain, ytest):

    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.01, max_depth=8)

    lgbm.fit(Xtrain, ytrain)

    # Predict on training and testing sets
    stacking_train_pred = lgbm.predict_proba(Xtrain)[:, 1]
    stacking_test_pred = lgbm.predict_proba(Xtest)[:, 1]

    # Calculate ROC AUC scores
    stacking_train_acc = roc_auc_score(ytrain, stacking_train_pred)
    stacking_test_acc = roc_auc_score(ytest, stacking_test_pred)

    print('Train AUC:', stacking_train_acc)
    print('Test AUC:', stacking_test_acc)

    # Save the model
    joblib.dump(lgbm, 'voting_amazon_review_model.pkl')

    # Load the model
    loaded_model = joblib.load('voting_amazon_review_model.pkl')
    predictions = loaded_model.predict(Xtest)

    return predictions

predictions = model_stacking(X_train_tf_idf_word, X_test_tf_idf_word, y_train, y_test)


def plot_roc_curve(y_test, y_pred_proba, save_path=None):

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()


plot_roc_curve(y_test, predictions, save_path='Model_ROC_Curve.png')