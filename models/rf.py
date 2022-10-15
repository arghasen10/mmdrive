from model_tuned import *


df = get_df()
X_train, X_test, y_train, y_test = get_xtrain_ytrain(df, frame_stack=10)
rfModel = rf_model(X_train, X_test, y_train, y_test)
rfModel.train()
test_result = rfModel.test()
print(test_result)
plot_confusion_mat(test_result)
