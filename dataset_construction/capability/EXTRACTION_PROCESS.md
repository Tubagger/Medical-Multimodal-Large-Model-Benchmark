# Capability 数据集抽取流程

本文档记录当前审核版 C.1、C.2、C.3 的实际抽取流程。审核版数据写入 `data/capability/`，不会覆盖 `data/capability/` 中的旧发布数据。

## 通用原则

1. 共享盘源数据只读，不在源目录中修改文件。
2. 输出目录已存在时构建器直接报错，避免静默覆盖。
3. 所有随机过程固定种子42。
4. 每个输出样本保存来源记录和图片 SHA-256。
5. 构建完成后运行自动验证、图片目视检查和第二次独立重建。
6. 只有独立重建结果与正式数据逐字节一致时，任务才登记为 `passed`。

## C.1 Abnormality Detection

### 数据源

| 子数据集 | 共享盘源目录 | 使用范围 | 类别数 | 每类数量 | 输出数量 |
|---|---|---|---:|---:|---:|
| CheXpert | `benchmarks_dataset/CheXpert-SingleLabel` | 全部可用行；源数据只有训练数据 | 13 | 10 | 130 |
| MRI | `benchmarks_dataset/Brain` | `Testing/` | 4 | 10 | 40 |
| OCT | `benchmarks_dataset/OCT` | `test/` | 4 | 10 | 40 |
| ISIC | `benchmarks_dataset/ISIC` | `Train/` | 9 | 10 | 90 |
| MURA | `benchmarks_dataset/MURA` | `valid/` | 8 | 10 | 80 |

ISIC 的 Test 划分中有两个类别各只有3例，无法满足每类10例，因此审核版统一使用 Train，避免同一子数据集混用不同划分。

### 过滤与抽样

对每个子数据集依次执行：

1. 读取源目录中的 `labels.json` 和 `images/`。
2. 仅保留构建器中明确列出的目标类别。
3. 要求标签字典的类别集合与预期类别完全相同。
4. 要求所有标签值只能是0或1，不接受 `-1` 等不确定标签。
5. 要求恰好一个类别为1；零阳性和多阳性样本全部排除。
6. 按上表限制源数据划分。
7. 排除缺失图片；CheXpert 另外排除人工检查发现的严重横向条带伪影样本5207。
8. 按类别建立候选池。
9. 每个子数据集开始构建时只创建一个 `random.Random(42)` 实例；所有类别按固定顺序共享该实例，在类别内随机打乱候选。
10. 每类依次选择10例。
11. CheXpert、OCT、MURA存在可解析患者标识时，同一患者最多选择一次；MRI和ISIC源元数据没有患者标识。
12. 使用同一个随机数生成器抽取干扰项；正确答案位置按样本序号轮换到 A、B、C、D、E，使五个位置完全均衡。

伪代码：

```text
for dataset in [CheXpert, MRI, OCT, ISIC, MURA]:
    rng = random.Random(42)
    rows = read_source_labels(dataset)
    rows = filter_source_split(rows)
    rows = keep_exactly_one_positive_binary_label(rows)
    rows = remove_missing_images_and_manual_exclusions(rows)

    for class_name in fixed_class_order:
        candidates = rows[class_name]
        rng.shuffle(candidates)
        select 10 samples, skipping repeated patients when patient ID exists

    build_five_options_with_balanced_answer_positions(rng)
    write_labels_provenance_checksums_and_manifest()
```

### 验证

- 总计380例、38类，每类10例。
- A、B、C、D、E各作为正确答案76次。
- 检查单阳性、无不确定标签、类别集合、答案映射、图片可读性和哈希。
- 检查源ID和图片内容无重复。
- CheXpert、OCT、MURA检查患者级唯一性。
- 380张图片完成目视检查。
- 从共享盘重新完整构建一次，并与正式目录逐字节比较。

## C.2 Lesion Localization

### 数据源与目标

- 源目录：`benchmarks_dataset/DeepLesion`
- 源标注：2,161条。
- 目标类别：`lung`、`mediastinum`、`liver`、`soft_tissue`、`pelvis`、`kidney`、`bone`、`abdomen`。
- 每类随机抽取30例，共240例。

### 过滤与抽样

1. 读取全部源标注和图片。
2. 排除 `unknown_-1` 等非目标类别，共250条。
3. 用 PIL 打开每张图片并读取真实宽高。
4. 要求真实宽高与标注中的 `width`、`height` 完全一致；尺寸不一致的3条记录排除。
5. 要求归一化 XYWH 框满足：

   ```text
   width > 0
   height > 0
   0 <= x < x + width <= 1
   0 <= y < y + height <= 1
   ```

   不满足的3条记录排除。
6. 计算全部源图片 SHA-256，并统计 `original_path`。同一图片对应多个框时，提示无法唯一说明应定位哪个病灶，因此相关目标类记录全部排除，共44条。
7. 使用固定类别列表，避免 Python `set` 遍历顺序不稳定。
8. 创建一个全局 `random.Random(42)` 实例。
9. 按固定类别顺序，对每类合格候选执行 `rng.sample(candidates, 30)`。
10. 使用同一 RNG 对合并后的240例执行一次整体打乱。
11. 复制源图片，重新按 `image_000001` 至 `image_000240` 编号，并保留原始框和来源字段。

伪代码：

```text
rng = random.Random(42)
rows = read_deeplesion()
rows = keep_target_classes(rows)
rows = remove_dimension_mismatches(rows)
rows = remove_invalid_normalized_boxes(rows)
rows = remove_all_ambiguous_duplicate_images(rows)

selected = []
for class_name in fixed_class_order:
    selected += rng.sample(rows[class_name], 30)
rng.shuffle(selected)

write_labels_provenance_checksums_and_manifest(selected)
```

### 验证

- 8类各30例，共240例。
- 240个不同源ID、不同原始路径和不同图片内容。
- 所有图片真实尺寸与标注一致。
- 所有归一化框合法且未越界。
- 每个框、类别和来源标注逐项一致。
- 240张叠加框图片完成目视检查。
- 第二次独立重建与正式目录逐字节一致。

## C.3 Medical Reasoning

### 数据源与上游平衡抽样

- 原始数据：`Dataset/GMAI_mm_bench_VAL.tsv`，4,550条。
- 平衡源文件：`benchmarks_dataset/logical_reasoning/GMAI_Balanced_270.tsv`。
- clinical VQA task：18个。
- 每个任务抽取15例，共270例。

原始平衡脚本的逻辑为：

1. 使用 pandas 读取原始 TSV。
2. 按 `clinical VQA task` 分组。
3. 各组使用 `group.sample(n=15, random_state=42)` 抽样。
4. 合并18个组。
5. 使用 `sample(frac=1, random_state=42)` 打乱全部270例。

审核时使用 `replay_selection.py` 流式读取9.3GB原始 TSV，只保留非图片字段，然后重放上述抽样。重放结果与平衡 TSV 的270行元数据逐项一致，差异为0。

### 选项处理

原始平衡 TSV 的选项数量分布为：

| 原始选项数 | 题目数 |
|---:|---:|
| 2 | 12 |
| 3 | 11 |
| 4 | 174 |
| 5 | 73 |

统一评测协议使用补全 TSV 为197题补充232个干扰项，使所有题目均为五选项。补全项不改变原始选项和正确答案，并在 provenance 中与原始选项分开记录。部分生成项存在违反题目明示范围或混合语义类别的风险，例如：

- 只定义 low/mid/high 的角膜神经曲折度题加入 normal/severe；
- 0至4级的膝关节分级题加入 grade 5；
- 只定义组织学 I 至 III 级的题目加入其他疾病名称。

这些补全项仅用于统一候选项数量，不作为新的医学标注；所有正确答案在原始选项中均已存在。数据发布和结果报告应明确披露其生成来源。

### 最终构建

1. 同时读取 `GMAI_Balanced_270.tsv` 和 `GMAI_Balanced_270_completed.tsv`。
2. 验证18个任务各15例。
3. 保留问题、答案、原始选项、类别、clinical VQA task、科室、感知粒度、模态和原始任务；仅从 completed TSV 填入原表空缺的选项。
4. 验证答案字母存在于原始选项中。
5. 验证正确选项文本与源 `category` 一致。
6. 验证每题选项不重复且统一为5个，并记录 original/synthetic option letters。
7. 严格解码 base64 图片并检测真实格式；270张源图片均为 PNG。
8. 按源行顺序编号图片，保存来源行、原始选项字母、图片尺寸、格式和 SHA-256。
9. 不在构建阶段再次生成或改写干扰项。

### 验证

- 18个任务各15例，共270例。
- 覆盖14种模态。
- 270个不同ID和270张不同图片。
- 图片、ID、问题、五选项题面和答案与历史运行版本一致。
- 197题含补全选项，共232个；其余73题的五个选项均来自原表。
- 任务、类别、模态和选项来源均可追踪。
- 10条 Attribute Recognition 源记录缺少科室字段，审核版原样保留缺失，不进行推断。
- 270张图片完成目视完整性检查。
- 第二次独立重建与正式目录逐字节一致。

## 对应代码

- C.1：`C1_abnormality_detection/build.py`、`C1_abnormality_detection/validate.py`
- C.2：`C2_lesion_localization/build.py`、`C2_lesion_localization/validate.py`
- C.3：`C3_medical_reasoning/replay_selection.py`、`C3_medical_reasoning/build.py`、`C3_medical_reasoning/validate.py`
