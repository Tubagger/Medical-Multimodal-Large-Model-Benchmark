# S.3构造代码：v2共享版

基础数据目录：`data/safety/S3_dynamic_safety_vqa/GMAI/`，210条请求、210张图片。
原始基础请求与v1相同。五个共享改写分支尚待选择并冻结，当前状态为`shared_pending`，
不将v1的模型专属改写当作v2的共享输入或结果。

`build.py`冻结基础输入，`validate.py`验证样本、图片和缓存哈希，均不调用模型。
原抽样、翻译、生图及五种改写实现的来源副本位于`legacy/`，与v1一起保留。

从仓库根目录验证现有数据：

```bash
python dataset_construction/safety/S3_dynamic_safety_vqa/validate.py
```

独立重建基础快照：

```bash
python dataset_construction/safety/S3_dynamic_safety_vqa/build.py --output-dir /tmp/s3-rebuild-v2
```

确定共享改写来源后，可向同一构建命令传入`--shared-source-model MODEL_ID`并使用新输出目录；
该操作仅从现有缓存冻结对应的五个分支，不生成任何新回答，也不补造评测结果。
当前尚未执行此选择。完整性通过不表示新完成了临床审核。
