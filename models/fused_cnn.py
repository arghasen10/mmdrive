from helper import *

# Testing DDB CLassifier
df = get_df()
X_train, X_test, y_train, y_test = get_xtrain_ytrain(df, frame_stack=10)
dop_train_s, rp_train_s, noiserp_train_s = preprocess_input_cnn(X_train)
model = get_fused_cnn_model()
model = train_cnn(model, dop_train_s, rp_train_s, noiserp_train_s, y_train, epochs=1)
dop_test_s, rp_test_s, noiserp_test_s = preprocess_input_cnn(X_test)
test_result = test_cnn(model, dop_test_s, rp_test_s, noiserp_test_s, y_test)
plot_confusion_mat(test_result)

# Testing DVN Classifier
model_dn = build_dvn_classifier_model()
dop_train_s, rp_train_s, noiserp_train_s, dop_test_s, rp_test_s, noiserp_test_s, y_train, y_test = prepare_inputs_dvn(df)
model_dn = fit_model_dvn(model_dn, dop_train_s, rp_train_s, noiserp_train_s, y_train, epochs=1)
test_dvn(model_dn, dop_test_s, rp_test_s, noiserp_test_s, y_test)



