# DPMConformance
1.实验代码在code中，assignconformancevariant为分布式合规性计算，hrdecomposetreela为流程模型分解， original_method为原方法。

2.spark平台提交示例如下： bin/spark-submit --master spark://master:7077 --executor-memory 1G /home/hadoop/Projects/ProcessTreeDcompose/assignconformancevariant.py /home/hadoop/Projects/testdata/L1/L1.ptml hdfs://master:9000/L1.txt 4 2 alignment None 其中，/home/hadoop/Projects/ProcessTreeDcompose/assignconformancevariant.py是运行的代码文件，/home/hadoop/Projects/testdata/L1/L1.ptml是本地流程模型文件，hdfs://master:9000/L1.txt是轨迹字符串形式的事件日志，4是分解参数，2是Partition参数，alignment是合规性检查算法参数。
