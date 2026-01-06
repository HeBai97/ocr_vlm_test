# 图片解析命令示例

本文档提供了三个图片解析工具的常用命令示例。所有工具都支持递归查找目录及子目录中的图片文件，并提供两种处理模式：
- **随机抽取模式** (`--type random`): 随机抽取指定数量的图片进行处理
- **全部处理模式** (`--type all`): 处理所有找到的图片

## 1. PaddleOCRVL 解析工具

### 命令 1: 随机抽取 1 张图片
```bash
python ./test_paddleocrvl_demo.py test_ocr --type random --number 1
```
**说明**: 从 `test_ocr` 目录（包括子目录）中随机抽取 1 张图片进行解析。

### 命令 2: 处理所有图片
```bash
python ./test_paddleocrvl_demo.py test_ocr --type all
```
**说明**: 递归处理 `test_ocr` 目录及所有子目录中的所有图片文件。

## 2. DotsOCR VLM 解析工具

### 命令 3: 随机抽取 10 张图片
```bash
python ./test_dotsocrvlm_demo.py test_ocr --type random --number 10
```
**说明**: 从 `test_ocr` 目录（包括子目录）中随机抽取 10 张图片进行解析。

### 命令 4: 处理所有图片
```bash
python ./test_dotsocrvlm_demo.py test_ocr --type all
```
**说明**: 递归处理 `test_ocr` 目录及所有子目录中的所有图片文件。

## 3. MinerU 解析工具

### 命令 5: 随机抽取 5 张图片
```bash
python ./test_mineru_image.py test_ocr --type random --number 5
```
**说明**: 从 `test_ocr` 目录（包括子目录）中随机抽取 5 张图片进行解析。

### 命令 6: 处理所有图片
```bash
python ./test_mineru_image.py test_ocr --type all
```
**说明**: 递归处理 `test_ocr` 目录及所有子目录中的所有图片文件。

## 参数说明

### 通用参数
- `--type random/all`: 处理类型
  - `random`: 随机抽取模式，需要配合 `--number` 使用
  - `all`: 处理所有图片（默认值）
- `--number N`: 指定处理的图片数量（仅在 `--type random` 时有效）

### PaddleOCRVL 专用参数
- `--backend`: 后端引擎类型（默认: `vllm-server`）
- `--server-url`: 服务器地址（默认: `http://127.0.0.1:8080/v1`）
- `--output-dir`: 输出目录（默认: `paddleocrvl_output`）

### DotsOCR VLM 专用参数
- `--use-hf`: 使用本地 HuggingFace 推理
- `--use-vllm`: 使用 vllm 服务推理（默认）
- `--ip`: vllm 服务 IP（默认: `127.0.0.1`）
- `--port`: vllm 服务端口（默认: `8000`）
- `--model-name`: vllm 模型名称（默认: `dotsocr-model`）
- `--output-dir`: 输出目录（默认: `dotsocr_output`）
- `--device`: GPU 设备（仅 `--use-hf` 时有效）

### MinerU 专用参数
- `--lang`: 语言设置（默认: `ch`）
- `--backend`: 后端引擎类型（默认: `vlm-http-client`）
- `--server-url`: 服务器地址（默认: `http://localhost:30000`）
- `--output-dir`: 输出目录（默认: `mineru_output`）

## 4. 统一OCR调用工具 (test_ocr_demo.py)

`test_ocr_demo.py` 是一个统一调用工具，可以依次调用上述三个OCR脚本，自动处理图片并生成统计报告。

### 命令 7: 随机抽取图片并统一处理
```bash
python ./test_ocr_demo.py test_ocr --type random --number 10
```
**说明**: 
- 从 `test_ocr` 目录中随机抽取 10 张图片
- 依次使用 PaddleOCRVL、DotsOCR、MinerU 三个工具进行解析
- 将标注后的图片保存到 `./output` 目录，文件名格式：`原文件名_ocr类型.扩展名`
- 生成统计报告 `./output/statistics_report.md`

### 命令 8: 处理所有图片并统一处理
```bash
python ./test_ocr_demo.py test_ocr --type all
```
**说明**: 
- 处理 `test_ocr` 目录中的所有图片
- 依次使用三个OCR工具进行解析
- 生成统计报告和标注图片

### 功能特点

1. **统一调用**: 自动调用三个OCR脚本，无需手动切换conda环境
2. **结果整理**: 
   - 标注图片统一保存到 `./output` 目录
   - 文件名格式：`图片名_paddle.jpg`、`图片名_dots.jpg`、`图片名_mineru.jpg`
3. **统计报告**: 自动生成Markdown格式的统计报告，包含：
   - 每张图片的解析元素个数
   - 每张图片的处理时间
   - 每个OCR工具的总计统计

### 输出文件结构

```
./output/
├── 图片名1_paddle.jpg      # PaddleOCRVL标注结果
├── 图片名1_dots.jpg         # DotsOCR标注结果
├── 图片名1_mineru.jpg      # MinerU标注结果
├── 图片名2_paddle.jpg
├── 图片名2_dots.jpg
├── 图片名2_mineru.jpg
└── statistics_report.md    # 统计报告
```

### 统计报告示例

统计报告包含三个表格，分别对应三个OCR工具：

```markdown
## PADDLE OCR 统计

| 图片名 | 解析出元素个数 | 处理时间(秒) |
|--------|---------------|-------------|
| image1.jpg | 15 | 2.50 |
| image2.jpg | 23 | 2.50 |
| **总计** | **38** | **5.00** |

## DOTS OCR 统计

| 图片名 | 解析出元素个数 | 处理时间(秒) |
|--------|---------------|-------------|
| image1.jpg | 18 | 3.20 |
| image2.jpg | 25 | 3.20 |
| **总计** | **43** | **6.40** |

## MINERU OCR 统计

| 图片名 | 解析出元素个数 | 处理时间(秒) |
|--------|---------------|-------------|
| image1.jpg | 16 | 4.10 |
| image2.jpg | 24 | 4.10 |
| **总计** | **40** | **8.20** |
```

### 参数说明

- `--type random/all`: 处理类型
  - `random`: 随机抽取模式，需要配合 `--number` 使用
  - `all`: 处理所有图片（默认值）
- `--number N`: 指定处理的图片数量（仅在 `--type random` 时有效，默认值：100）

### 注意事项

1. **Conda环境**: 脚本会自动切换到对应的conda环境执行OCR工具
2. **执行顺序**: 按照 PaddleOCRVL → DotsOCR → MinerU 的顺序依次执行
3. **输出目录**: 所有结果统一保存到 `./output` 目录
4. **处理时间**: 统计报告中的处理时间是平均分配到每张图片的（因为OCR工具是批量处理的）

## 注意事项

1. **目录路径**: 可以将 `test_ocr` 替换为任何包含图片的目录路径
2. **递归查找**: 所有工具都会递归查找指定目录及其子目录中的所有图片
3. **随机抽取**: 使用 `--type random` 时，必须指定 `--number` 参数且大于 0
4. **文件格式**: 支持的图片格式包括 `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tiff`, `.tif`（MinerU 还支持 `.pdf`）
5. **统一工具**: 使用 `test_ocr_demo.py` 时，确保三个conda环境（`paddleOCR`、`dots_ocr`、`mineru`）都已正确配置

