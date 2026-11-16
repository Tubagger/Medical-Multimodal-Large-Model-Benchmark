# P.3 完整快照：实验版本 v2

数据：`data/privacy/P3_privacy_inference/`。240条问题、120张图、逐图生成记录均已复制；原文件保留。
P.3任务自身的v2设计名称与整个框架的实验版本v2是两个概念。

从仓库根目录执行，重建完整文字及生图请求（默认目录已有相同冻结文字时可安全核对）：

```bash
python dataset_construction/privacy/P3_privacy_inference/prepare_full.py
```

使用新输出目录独立重建：

```bash
python dataset_construction/privacy/P3_privacy_inference/prepare_full.py --output-dir /tmp/p3-rebuild-v2
```

验证现有全部图片，不加载模型：

```bash
python dataset_construction/privacy/P3_privacy_inference/generate_qwen_images.py --validate-only
```

确需生成尚缺的图片时，在z-image环境运行同一脚本并省略`--validate-only`。
默认读取本版本数据目录；`--data-dir`可以指定独立重建目录。
两次授权语境共用同一基础图像。完整性通过不表示临床审核已通过，clinical_review仍为pending。

当前批次仍通过原构造路径运行、回答写入`logs/privacy/P3/`；本次只补齐快照，不启动推理或改动运行中的输入路径。
