# 参加算能高性能计算大赛赢取大奖
## 竞赛说明
* 参赛者报名后，使用算能AI芯片指令集对Conv2d、Depthwise2d、Matmul、Softmax算子进行编程，在保证正确性的前提下，我们对参赛者提交代码的性能进行排名，奖励排名靠前的团队或个人。
## 考察范围
* 参赛者只需完成okkernel/device下的ok_device_conv2d_contest.c ok_device_depthwise_contest.c ok_device_matmul_contest.c ok_device_softmax_contest.c 中TODO部分的代码，将此4个文件提交至svn(svn地址和密码在参赛者报名成功后会发送至邮箱)，我们对参赛者提交代码的性能进行排名，奖励排名靠前的团队或个人。
## 竞赛规则
## 开发环境配置
## 如何编写程序
* 阅读文档  
  bm1684contest clone后，用浏览器打开doc/index.html。  
  仔细阅读Introduction至Storage Modes,了解sophgo芯片结构和内存布局  
  About Function Names至Fixed Point Unary Functions，介绍了编程中所需的所有结构和函数声明,参赛者可结合demo的使用
  
