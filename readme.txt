操作步骤：
1.运行process1.py，根据1101_score_1_1_result.xlsx，1101_score_1_2_result.xlsx，1101_score_1_3_result.xlsx中的图片及标注
生成分行后的训练数据，图片输出到train_img文件夹中，标注文本输出到train_txt文件夹中（需要有网）；
2.运行process2.py，根据1101_score_1_4_result.xlsx中的图片及标注生成测试数据，图片输出到test_img中，标注文本输出到
test_txt中（需要有网）；
3.运行train_model.py进行训练；
4.运行test_model.py进行预测。

运行环境：
1.Python2、3均可；
2.tensorflow1.2.1及以上