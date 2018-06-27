Predicting user's demographic information and personality through their browsing history

Data pre-processing:
1.  all_user_csv_out_2.py (browsing history to web categories)
2. update_all_user_v3.py (merge some web categories)
3. user_daily_v4.py (output users feature with categories ratio)
4. user_daily_v5.py (output users feature with time session frequency)

Demographic information prediction:
1. supervise_demo_KNN.py (predicting user's demographic information in k-NN)
2. supervise_demo_RF.py (predicting user's demographic information in random forests)
3. supervise_demo_LR.py (predicting user's demographic information in logistic regression)
4. supervise_demo_SVM.py (predicting user's demographic information in SVM)
5. kms_demo_KNN.py (predicting user's demographic information in clustering with k-NN)
6. kms_demo_RF.py (predicting user's demographic information in clustering with random forests)
7. kms_demo_LR.py (predicting user's demographic information in clustering with logistic regression)
8. kms_demo_SVM.py (predicting user's demographic information in clustering with SVM)

Big-six personality prediction:
1. supervise_pr_SVM.py (predicting user's big-six personality in SVR) 
2. supervise_pr_Lasso.py (predicting user's big-six personality in Lasso regression) 
3. supervise_pr_Ridge.py (predicting user's big-six personality in Ridge regression) 
4. supervise_pr_EN.py (predicting user's big-six personality in Elastic net regression)
5. kms_pr_pred.py (predicting user's big-six personality in clustering with some regression models)


