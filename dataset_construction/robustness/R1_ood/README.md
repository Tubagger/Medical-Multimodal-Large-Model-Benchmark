# R.1 OOD Medical VQA Construction

本目录重构论文中的 R.1 OOD 医学 VQA 数据。构建器只接受冻结的
`GMAI_OOD_modality.tsv`，统一模态名称中的空白后，按模态分别使用
`numpy.random.RandomState(42)` 无放回抽取至多25例。该设置精确复现18个
模型日志共同使用的219个ID。

## 构建

```bash
python dataset_construction/robustness/R1_ood/build.py \
  --source-tsv /path/to/GMAI_OOD_modality.tsv \
  --runtime-labels data/robustness/ood/labels.json \
  --source-sha256 32d86dc1738c28c060ea796d945e53d85118f707167cae841e9cc8e35a58bec6 \
  --output-root data/robustness/R1_ood
```

构建器保留 GMAI-MMBench 原始选项，并从冻结的运行标签恢复后补选项，最终
219题均为五选一。样本级 provenance 分开记录原始与后补选项字母。该版本
复现日志实际输入，不重新调用生成模型。题目与选项来自已发表 benchmark 的
冻结发布版本，以其发布标注为权威来源，不要求本项目再次进行临床审核。

## 验证

```bash
python dataset_construction/robustness/R1_ood/validate.py \
  data/robustness/R1_ood/GMAI
```

验证覆盖题量、25题模态上限、日志ID顺序、原始选项未被改写、答案与类别
映射、图片格式和尺寸、运行图片逐字节一致、逐文件 SHA-256、样本级来源及
清单一致性。

源过滤可单独复放：

```bash
python dataset_construction/robustness/R1_ood/replay_source_filter.py \
  --raw-tsv /path/to/GMAI_mm_bench_VAL.tsv \
  --ood-tsv /path/to/GMAI_OOD_modality.tsv
```

上游 GMAI-MMBench 数据仓库声明 Apache-2.0，但数据访问受门控条件约束；
使用者仍须遵守上游访问条件及其汇集来源的适用条款。
