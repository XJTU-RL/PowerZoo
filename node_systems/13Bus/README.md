# IEEE13 运行输入

`IEEE13Node_BusXY.csv` 是本目录 DSS 文件通过 `BusCoords` 引用的母线绘图坐标。
缺少该文件会使 DSS-Python 在编译模型时抛出错误 #275，影响实际环境创建。
该文件恢复自 OpenDSS 的 IEEE13 标准测试算例，不改变线路、负荷或电气参数。

来源：[DSS-Extensions 的 OpenDSS 测试数据镜像](https://github.com/dss-extensions/electricdss-tst/blob/master/Version8/Distrib/IEEETestCases/13Bus/IEEE13Node_BusXY.csv)，
恢复日期为 2026-10-27。坐标亦与
[OpenDSS 历史源码镜像](https://github.com/tshort/OpenDSS/blob/master/Distrib/IEEETestCases/13Bus/IEEE13Node_BusXY.csv)
一致。
