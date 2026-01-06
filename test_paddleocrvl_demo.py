import json
import os
import pathlib
import sys
import random
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCRVL
from typing import List


def draw_bboxes_from_paddleocrvl(paddleocrvl_result, image_path: str, output_path: str = None):
    """
    从PaddleOCRVL解析结果，在图片上绘制标注框
    
    Args:
        paddleocrvl_result: PaddleOCRVL解析返回的结果对象
        image_path: 输入图片路径
        output_path: 输出图片路径，如果为None则自动生成
    """
    # 打开原始图片
    image = Image.open(image_path).convert("RGB")
    
    # 定义颜色映射（不同类别使用不同颜色）
    category_colors = {
        'text': (255, 0, 0),           # 红色
        'title': (0, 255, 0),          # 绿色
        'number': (0, 0, 255),         # 蓝色
        'caption': (255, 165, 0),      # 橙色
        'figure': (255, 255, 0),       # 黄色
        'table': (255, 0, 255),        # 紫色
        'formula': (0, 255, 255),      # 青色
        'header': (128, 128, 128),     # 灰色
        'footer': (128, 128, 128),     # 灰色
    }
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            font = ImageFont.load_default()
    
    # 从PaddleOCRVL结果中提取parsing_res_list
    parsing_res_list = []
    
    # 方法1: 直接从结果对象获取
    if hasattr(paddleocrvl_result, 'parsing_res_list'):
        parsing_res_list = paddleocrvl_result.parsing_res_list
    
    # 方法2: 通过保存JSON然后读取
    if not parsing_res_list:
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                paddleocrvl_result.save_to_json(save_path=temp_dir)
                json_files = list(pathlib.Path(temp_dir).glob("*.json"))
                if json_files:
                    with open(json_files[0], 'r', encoding='utf-8') as f:
                        result_dict = json.load(f)
                    parsing_res_list = result_dict.get('parsing_res_list', [])
        except Exception as e:
            print(f"无法从 PaddleOCRVL 结果中提取 parsing_res_list: {e}")
            return None, None
    
    if not parsing_res_list:
        print("未找到解析结果")
        return None, None
    
    # 创建绘制对象
    draw = ImageDraw.Draw(image)
    
    print(f"找到 {len(parsing_res_list)} 个文本块，开始绘制标注框...")
    
    # 绘制每个block的bbox
    for idx, block in enumerate(parsing_res_list):
        if not isinstance(block, dict):
            continue
        
        block_label = block.get('block_label', 'text').lower()
        block_content = block.get('block_content', '').strip()
        block_bbox = block.get('block_bbox', [])
        
        if len(block_bbox) == 4:
            x1, y1, x2, y2 = [int(coord) for coord in block_bbox]
            
            # 获取颜色
            color = category_colors.get(block_label, (128, 128, 128))
            
            # 绘制半透明填充矩形
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=color + (50,))
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # 绘制边框（更粗一些以便清晰可见）
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签文本（在bbox左上角）
            label_text = f"{idx+1}:{block_label}"
            # 计算文本位置（确保不超出图片边界）
            text_x = max(5, min(x1 + 5, image.width - 200))
            text_y = max(5, min(y1 - 30, image.height - 30))
            
            # 绘制文本背景（白色半透明）
            try:
                bbox_text = draw.textbbox((text_x, text_y), label_text, font=font)
                # 创建半透明背景层
                text_overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                text_overlay_draw = ImageDraw.Draw(text_overlay)
                text_overlay_draw.rectangle(bbox_text, fill=(255, 255, 255, 220))
                image = Image.alpha_composite(image.convert('RGBA'), text_overlay).convert('RGB')
                draw = ImageDraw.Draw(image)
                draw.text((text_x, text_y), label_text, fill=color, font=font)
            except:
                # 如果字体不支持，使用默认字体
                bbox_text = draw.textbbox((text_x, text_y), label_text)
                # 创建半透明背景层
                text_overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                text_overlay_draw = ImageDraw.Draw(text_overlay)
                text_overlay_draw.rectangle(bbox_text, fill=(255, 255, 255, 220))
                image = Image.alpha_composite(image.convert('RGBA'), text_overlay).convert('RGB')
                draw = ImageDraw.Draw(image)
                draw.text((text_x, text_y), label_text, fill=color)
    
    # 保存结果
    if output_path is None:
        image_name = pathlib.Path(image_path).stem
        output_path = f"{image_name}_annotated.jpg"
    
    # 确保输出目录存在
    output_dir = pathlib.Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    image.save(output_path, "JPEG", quality=95)
    print(f"标注后的图片已保存到: {output_path}")
    
    return image, output_path


def find_image_files_recursive(directory: str, image_extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')) -> List[str]:
    """
    递归查找目录下所有图片文件
    
    Args:
        directory: 要搜索的目录路径
        image_extensions: 图片文件扩展名元组
        
    Returns:
        List[str]: 所有找到的图片文件路径列表
    """
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files


def parse_image_with_paddleocrvl(image_path: str,
                                  pipeline: PaddleOCRVL = None,
                                  vl_rec_backend: str = "vllm-server",
                                  vl_rec_server_url: str = "http://127.0.0.1:8080/v1",
                                  output_dir: str = "paddleocrvl_output"):
    """
    使用 PaddleOCRVL 解析图片并生成标注图片
    
    Args:
        image_path: 输入图片路径
        pipeline: 可选的 PaddleOCRVL 实例，如果提供则复用（避免重复初始化）
        vl_rec_backend: 视觉识别后端类型，默认为 "vllm-server"
        vl_rec_server_url: 视觉识别服务器地址，默认为 "http://127.0.0.1:8080/v1"
        output_dir: 输出目录，默认为 "paddleocrvl_output"
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有提供 pipeline，则创建一个新的
    if pipeline is None:
        pipeline = PaddleOCRVL(vl_rec_backend=vl_rec_backend, vl_rec_server_url=vl_rec_server_url)
    
    print(f"解析图片: {image_path}")
    
    try:
        output = pipeline.predict(image_path)
    except Exception as e:
        print(f"解析失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    if not output:
        print("解析失败，未返回结果")
        return None
    
    image_name = pathlib.Path(image_path).stem
    
    # 处理每个结果
    for idx, res in enumerate(output):
        res.print()
        
        # 保存 JSON 和 Markdown
        res.save_to_json(save_path=output_dir)
        res.save_to_markdown(save_path=output_dir)
        
        # 绘制标注框（如果有多个结果，使用索引区分）
        if len(output) > 1:
            annotated_path = os.path.join(output_dir, f"{image_name}_annotated_{idx+1}.jpg")
        else:
            annotated_path = os.path.join(output_dir, f"{image_name}_annotated.jpg")
        
        annotated_image, annotated_path_result = draw_bboxes_from_paddleocrvl(res, image_path, output_path=annotated_path)
        if annotated_image is None or annotated_path_result is None:
            print(f"警告: 无法为图片 {image_path} 生成标注图片（可能没有解析到内容）")
    
    print(f"解析完成，结果已保存到: {output_dir}")
    
    return {
        'output': output,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    # 默认配置
    vl_rec_backend = "vllm-server"
    vl_rec_server_url = "http://127.0.0.1:8080/v1"
    output_dir = "paddleocrvl_output"
    
    # 默认处理 bookpic 目录中的所有图片
    images_dir = "bookpic"
    
    # 解析命令行参数
    args_list = sys.argv[1:]
    i = 0
    number_arg = None
    type_arg = "all"  # 默认处理所有图片
    
    while i < len(args_list):
        if args_list[i] == "--backend" and i + 1 < len(args_list):
            vl_rec_backend = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--server-url" and i + 1 < len(args_list):
            vl_rec_server_url = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--output-dir" and i + 1 < len(args_list):
            output_dir = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--number" and i + 1 < len(args_list):
            number_arg = int(args_list[i + 1])
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--type" and i + 1 < len(args_list):
            type_arg = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        i += 1
    
    if len(args_list) > 0:
        # 如果提供了命令行参数，使用指定的图片或目录
        input_path = args_list[0]
        if os.path.isfile(input_path):
            # 单个图片文件
            parse_image_with_paddleocrvl(
                input_path,
                vl_rec_backend=vl_rec_backend,
                vl_rec_server_url=vl_rec_server_url,
                output_dir=output_dir
            )
        elif os.path.isdir(input_path):
            # 目录
            images_dir = input_path
        else:
            print(f"错误: {input_path} 不是有效的文件或目录")
            sys.exit(1)
    
    # 处理 images_dir 目录中的所有图片
    if os.path.isdir(images_dir):
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')
        
        # 递归查找所有图片文件
        all_image_files = find_image_files_recursive(images_dir, image_extensions)
        
        if not all_image_files:
            print(f"在 {images_dir} 目录中未找到图片文件")
            sys.exit(1)
        
        # 根据 --type 参数决定处理方式
        if type_arg == "random":
            # 随机抽取模式
            if number_arg is None or number_arg <= 0:
                print("错误: --type random 需要指定 --number 参数且大于0")
                sys.exit(1)
            if number_arg > len(all_image_files):
                print(f"警告: 请求数量 {number_arg} 大于找到的图片数量 {len(all_image_files)}，将处理所有图片")
                image_files = all_image_files
            else:
                image_files = random.sample(all_image_files, number_arg)
                print(f"随机抽取 {len(image_files)} 张图片（共找到 {len(all_image_files)} 张）")
        elif type_arg == "all":
            # 处理所有图片
            image_files = all_image_files
            print(f"找到 {len(image_files)} 张图片，开始处理所有图片...")
        else:
            # 默认处理所有图片
            image_files = all_image_files
            print(f"找到 {len(image_files)} 张图片，开始处理所有图片...")
        
        print(f"实际处理 {len(image_files)} 张图片")
        print(f"后端引擎: {vl_rec_backend}")
        print(f"服务器地址: {vl_rec_server_url}")
        print(f"输出目录: {output_dir}")
        
        # 创建 pipeline 实例（只初始化一次，复用）
        print("\n正在初始化 PaddleOCRVL 解析器（仅初始化一次，后续图片将复用）...")
        pipeline = PaddleOCRVL(vl_rec_backend=vl_rec_backend, vl_rec_server_url=vl_rec_server_url)
        print("解析器初始化完成，开始处理图片...\n")
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                image_file = os.path.relpath(image_path, images_dir)
            except ValueError:
                image_file = image_path
            print(f"\n[{idx}/{len(image_files)}] 处理: {image_file}")
            try:
                # 传入 pipeline 实例，避免重复初始化
                parse_image_with_paddleocrvl(
                    image_path,
                    pipeline=pipeline,  # 复用同一个 pipeline 实例
                    vl_rec_backend=vl_rec_backend,
                    vl_rec_server_url=vl_rec_server_url,
                    output_dir=output_dir
                )
            except Exception as e:
                print(f"处理 {image_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n所有图片处理完成！")
    else:
        print(f"错误: {images_dir} 目录不存在")
        sys.exit(1)