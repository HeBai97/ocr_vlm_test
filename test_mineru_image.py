#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 MinerU 进行图片解析和标注
支持使用 vlm-http-client 后端通过 OpenAI 兼容的服务器进行解析

使用方法:
    python test_mineru_image.py [图片路径或目录] --lang ch --backend vlm-http-client --server-url http://localhost:30000

参数说明:
    --lang: 语言设置，默认为 "ch"
    --backend: 后端引擎类型，默认为 "vlm-http-client"
    --server-url: OpenAI 兼容的服务器地址，默认为 "http://localhost:30000"
    --output-dir: 输出目录，默认为 "mineru_output"
"""

import re
import json
import pathlib
import os
import copy
import sys
import random
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# MinerU 相关导入
try:
    import asyncio
    from pathlib import Path
    from mineru.cli.common import aio_do_parse, read_fn, prepare_env
except ImportError as e:
    print(f"导入 MinerU 模块失败: {e}")
    print("\n请确保已正确安装 magic-pdf:")
    print("  pip install magic-pdf")
    sys.exit(1)

# 尝试导入 fitz (PyMuPDF) 用于图片转 PDF
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("警告: 未安装 PyMuPDF (fitz)，图片转 PDF 功能可能不可用")


def _is_latin_start(text: str) -> bool:
    """判断文本是否以拉丁字母开头"""
    try:
        if len(text) == 0:
            return False
        return text[0].isalpha()
    except Exception as e:
        print(text)
        raise e


def _merge_all_lines_on_block(block: dict, tag: str = 'content') -> str:
    """
    合并块中的所有行文本
    
    Args:
        block: 包含 lines 的块字典
        tag: 要提取的标签，'content' 或 'html'
        
    Returns:
        str: 合并后的文本
    """
    lines = block.get('lines', [])
    res = ''
    for line in lines:
        spans = line.get('spans', [])
        for span in spans:
            cur_text = span.get(tag, '')
            if _is_latin_start(cur_text):
                if len(res) > 0 and res[-1] == '-':
                    res = res[:-1] + cur_text
                else:
                    res += ' ' + cur_text
            else:
                res += cur_text
    return res.lstrip()


def parse_mineru_output(middle_res: List[dict]) -> List[dict]:
    """
    解析 MinerU 的输出结果，以二级块类型展示
    包括 para_blocks/preproc_blocks 和 discarded_blocks（如页码、页眉、页脚等）
    
    参考: https://opendatalab.github.io/MinerU/reference/output_files/#level-2-block-fields
    
    Args:
        middle_res: MinerU 解析后的中间结果，包含 pdf_info 列表
        
    Returns:
        List[dict]: 按二级块类型组织的列表
    """
    cells = []
    
    for page in middle_res:
        # MinerU 可能使用 para_blocks 或 preproc_blocks
        chunks = page.get('para_blocks', []) or page.get('preproc_blocks', [])
        
        for ck in chunks:
            # 跳过已删除的块
            if ck.get('lines_deleted', False):
                continue
            
            # 检查是否有二级块（Level 2 blocks）
            level2_blocks = ck.get('blocks', [])
            
            if level2_blocks:
                # 如果有二级块，提取所有二级块
                for block in level2_blocks:
                    block_type = block.get('type', '')
                    block_bbox = block.get('bbox', [])
                    
                    # 只处理有效的二级块类型
                    if block_type in ['image_body', 'image_caption', 'image_footnote',
                                     'table_body', 'table_caption', 'table_footnote',
                                     'text', 'title', 'index', 'list', 'interline_equation']:
                        if len(block_bbox) == 4:
                            # 提取文本内容
                            text = _merge_all_lines_on_block(block)
                            cells.append({
                                'type': block_type,
                                'bbox': block_bbox,
                                'text': text.strip()
                            })
            else:
                # 如果没有二级块，使用一级块（但需要转换为二级块类型）
                ck_type = ck.get('type', 'text')
                ck_bbox = ck.get('bbox', [])
                
                if len(ck_bbox) == 4:
                    # 将一级块类型映射到二级块类型
                    # 根据文档，一级块类型需要映射到对应的二级块类型
                    level2_type = ck_type
                    
                    # 如果一级块类型不在二级块类型列表中，保持原样或映射
                    if ck_type == 'image':
                        # image 类型的一级块，如果没有二级块，可能是 image_body
                        level2_type = 'image_body'
                    elif ck_type == 'table':
                        # table 类型的一级块，如果没有二级块，可能是 table_body
                        level2_type = 'table_body'
                    elif ck_type not in ['image_body', 'image_caption', 'image_footnote',
                                        'table_body', 'table_caption', 'table_footnote',
                                        'text', 'title', 'index', 'list', 'interline_equation']:
                        # 如果类型不在二级块类型列表中，保持原样
                        level2_type = ck_type
                    
                    # 提取文本内容
                    text = _merge_all_lines_on_block(ck)
                    cells.append({
                        'type': level2_type,
                        'bbox': ck_bbox,
                        'text': text.strip()
                    })
        
        # 处理 discarded_blocks（包括页码、页眉、页脚等）
        discarded_blocks = page.get('discarded_blocks', [])
        for discarded_block in discarded_blocks:
            # 跳过已删除的块
            if discarded_block.get('lines_deleted', False):
                continue
            
            block_type = discarded_block.get('type', '')
            block_bbox = discarded_block.get('bbox', [])
            
            # 处理页码、页眉、页脚等类型
            if block_type in ['page_number', 'header', 'footer', 'aside_text']:
                if len(block_bbox) == 4:
                    # 提取文本内容
                    text = _merge_all_lines_on_block(discarded_block)
                    cells.append({
                        'type': block_type,
                        'bbox': block_bbox,
                        'text': text.strip()
                    })
    
    return cells


def draw_bboxes_from_cells(cells: List[dict], image_path: str, output_path: str = None):
    """
    从 MinerU 解析的 cells 结果，在图片上绘制标注框
    适配原始 MinerU 数据格式
    
    Args:
        cells: MinerU 原始解析结果列表
        image_path: 输入图片路径
        output_path: 输出图片路径，如果为None则自动生成
    """
    # 打开原始图片
    image = Image.open(image_path).convert("RGB")

    # 定义颜色映射（二级块类型）
    # 参考: https://opendatalab.github.io/MinerU/reference/output_files/#level-2-block-types
    category_colors = {
        'text': (255, 0, 0),                    # 红色 - Text block
        'title': (0, 255, 0),                    # 绿色 - Title block
        'index': (0, 0, 255),                    # 蓝色 - Index block
        'list': (0, 0, 255),                     # 蓝色 - List block
        'interline_equation': (0, 255, 255),     # 青色 - Interline formula block
        'image_body': (255, 255, 0),             # 黄色 - Image body
        'image_caption': (255, 165, 0),          # 橙色 - Image caption text
        'image_footnote': (255, 192, 203),       # 粉色 - Image footnote
        'table_body': (255, 0, 255),             # 紫色 - Table body
        'table_caption': (128, 0, 128),           # 深紫色 - Table caption text
        'table_footnote': (148, 0, 211),         # 紫罗兰 - Table footnote
        'page_number': (0, 255, 128),            # 青绿色 - Page number
        'header': (255, 200, 0),                 # 金黄色 - Header
        'footer': (200, 200, 0),                 # 黄绿色 - Footer
        'aside_text': (128, 128, 128),           # 灰色 - Aside text
    }

    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            font = ImageFont.load_default()

    # 确定输出路径
    if output_path is None:
        image_name = pathlib.Path(image_path).stem
        output_path = f"{image_name}_annotated.jpg"
    
    # 确保输出目录存在
    output_dir = pathlib.Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果没有解析结果，直接保存原图
    if not cells:
        print("未找到解析结果，保存原图作为标注图片")
        image.save(output_path, "JPEG", quality=95)
        print(f"原图已保存到: {output_path}")
        return image, output_path

    # 创建绘制对象
    draw = ImageDraw.Draw(image)

    print(f"找到 {len(cells)} 个文本块，开始绘制标注框...")

    # 绘制每个cell的bbox
    for idx, cell in enumerate(cells):
        # 处理cell可能是字符串的情况
        if isinstance(cell, str):
            try:
                cell = json.loads(cell)
            except:
                continue

        if not isinstance(cell, dict):
            continue

        # 使用二级块类型（type 字段）
        block_type = cell.get('type', cell.get('category', 'text'))
        bbox = cell.get('bbox', [])

        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]

            # 获取颜色（使用二级块类型）
            color = category_colors.get(block_type, (128, 128, 128))

            # 绘制半透明填充矩形
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=color + (50,))
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(image)

            # 绘制边框（更粗一些以便清晰可见）
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 绘制标签文本（在bbox左上角，显示二级块类型）
            label_text = f"{idx+1}:{block_type}"
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
    image.save(output_path, "JPEG", quality=95)
    print(f"标注后的图片已保存到: {output_path}")

    return image, output_path


def resize_image_if_needed(image: Image.Image, max_size: int = 2048) -> Image.Image:
    """
    如果图片尺寸超过限制，则按比例缩放
    
    Args:
        image: PIL Image 对象
        max_size: 最大尺寸（宽或高的最大值）
        
    Returns:
        Image.Image: 缩放后的图片
    """
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    
    # 按比例缩放
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    
    print(f"图片尺寸 {width}x{height} 超过限制，缩放为 {new_width}x{new_height}")
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def image_to_pdf_bytes(image_path: str) -> bytes:
    """
    将图片转换为单页 PDF 的二进制数据
    
    Args:
        image_path: 图片路径
        
    Returns:
        bytes: PDF 二进制数据
    """
    if not HAS_FITZ:
        raise ImportError("需要安装 PyMuPDF (pip install pymupdf) 才能处理图片文件")
    
    # 使用 PyMuPDF 将图片转换为 PDF
    try:
        # 读取图片
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # 确定文件类型
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext == '.png':
            filetype = "png"
        elif file_ext in ['.jpg', '.jpeg']:
            filetype = "jpg"
        else:
            # 尝试作为图片打开
            filetype = None
        
        # 使用 fitz 将图片转换为 PDF
        if filetype:
            image_doc = fitz.open(stream=image_bytes, filetype=filetype)
        else:
            # 尝试自动识别
            image_doc = fitz.open(stream=image_bytes)
        
        pdf_bytes = image_doc.convert_to_pdf()
        image_doc.close()
        
        return pdf_bytes
    except Exception as e:
        print(f"图片转 PDF 失败: {e}")
        raise


class MinerUImageParser:
    """
    使用 MinerU 进行图片解析
    """
    
    def __init__(self, lang: str = "ch", backend: str = "vlm-http-client", server_url: str = "http://localhost:30000"):
        """
        初始化 MinerU 解析器
        
        Args:
            lang: 语言设置（"ch" 或 "en"）
            backend: 后端引擎类型，默认为 "vlm-http-client"
            server_url: OpenAI 兼容的服务器地址，默认为 "http://localhost:30000"
        """
        self.lang = lang
        self.backend = backend
        self.server_url = server_url
        print(f"初始化 MinerU 图片解析器，语言设置: {lang}, 后端: {backend}, 服务器地址: {server_url}")
    
    async def parse_image_async(self, image_path: str, output_dir: str = "./temp_output") -> tuple:
        """
        异步解析单张图片（使用 aio_do_parse）
        
        Args:
            image_path: 图片路径
            output_dir: 输出目录，默认为 "./temp_output"
            
        Returns:
            tuple: (cells, middle_json) - 解析结果和中间结果 JSON
        """
        print(f"解析图片: {image_path}")
        
        # 使用 read_fn 读取文件（自动处理图片转 PDF）
        try:
            pdf_bytes = read_fn(image_path)
        except Exception as e:
            print(f"读取文件失败: {e}")
            raise
        
        # 生成唯一的文件名
        import time
        file_name = f"{Path(image_path).stem}_{int(time.time())}"
        
        # 根据 backend 确定 parse_method
        if self.backend.startswith("vlm-"):
            parse_method = "vlm"
        else:
            parse_method = 'ocr' if self.lang not in ['ch', 'en'] else 'auto'
        
        # 准备输出目录
        if self.backend.startswith("hybrid"):
            env_name = f"hybrid_{parse_method}"
        else:
            env_name = parse_method
        
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, env_name)
        
        # 使用 aio_do_parse 进行解析
        try:
            await aio_do_parse(
                output_dir=output_dir,
                pdf_file_names=[file_name],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[self.lang],
                backend=self.backend,
                parse_method=parse_method,
                formula_enable=True,
                table_enable=True,
                server_url=self.server_url,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=False,
                f_dump_middle_json=True,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
            )
        except Exception as e:
            print(f"解析失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 读取 middle.json 文件
        middle_json_path = os.path.join(local_md_dir, f"{file_name}_middle.json")
        if not os.path.exists(middle_json_path):
            raise FileNotFoundError(f"未找到 middle.json 文件: {middle_json_path}")
        
        with open(middle_json_path, 'r', encoding='utf-8') as f:
            middle_json_data = json.load(f)
        
        # 转换为 JSON 字符串
        middle_json = json.dumps(middle_json_data, ensure_ascii=False)
        
        # 解析 MinerU 输出
        middle_res = middle_json_data.get('pdf_info', [])
        cells = parse_mineru_output(middle_res)
        
        return cells, middle_json
    
    def parse_image(self, image_path: str, output_dir: str = "./temp_output") -> tuple:
        """
        解析单张图片（同步包装器）
        
        Args:
            image_path: 图片路径
            output_dir: 输出目录，默认为 "./temp_output"
            
        Returns:
            tuple: (cells, middle_json) - 解析结果和中间结果 JSON
        """
        return asyncio.run(self.parse_image_async(image_path, output_dir))


def find_image_files_recursive(directory: str, image_extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.pdf')) -> List[str]:
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


def parse_image_with_annotation(image_path: str, 
                                 parser: MinerUImageParser = None,
                                 lang: str = "ch",
                                 backend: str = "vlm-http-client",
                                 server_url: str = "http://localhost:30000",
                                 output_dir: str = "mineru_output"):
    """
    解析图片并生成标注图片
    
    Args:
        image_path: 输入图片路径
        parser: 可选的 MinerUImageParser 实例，如果提供则复用（避免重复初始化）
        lang: 语言设置（仅在 parser 为 None 时使用）
        backend: 后端引擎类型，默认为 "vlm-http-client"
        server_url: OpenAI 兼容的服务器地址，默认为 "http://localhost:30000"
        output_dir: 输出目录，默认为 "mineru_output"
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有提供 parser，则创建一个新的
    if parser is None:
        parser = MinerUImageParser(lang=lang, backend=backend, server_url=server_url)
    
    print(f"解析图片: {image_path}")
    
    # 解析图片（使用临时输出目录，然后读取结果）
    temp_output_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_output_dir, exist_ok=True)
    
    try:
        cells, middle_json = parser.parse_image(image_path, output_dir=temp_output_dir)
    except (ImportError, TypeError) as e:
        # 这些是兼容性错误，已经在上层打印了详细的解决方案
        print(f"\n解析失败: {e}")
        print("请按照上述提示解决依赖问题后重试。")
        return None
    except Exception as e:
        error_msg = str(e)
        print(f"\n解析失败: {error_msg}")
        
        # 检查是否是 MFR 模型相关错误
        if 'MFR' in error_msg or 'mfr' in error_msg or 'UnimerMBart' in error_msg:
            print("\n提示: 这是数学公式识别（MFR）模型的错误。")
            print("如果图片中没有公式，可以忽略此错误。")
            print("如果需要公式识别功能，请检查 transformers 版本兼容性。")
        
        import traceback
        traceback.print_exc()
        return None
    
    if not cells:
        print("解析失败，未返回结果")
        return None
    
    print(f"解析完成，共找到 {len(cells)} 个文本块")
    
    image_name = pathlib.Path(image_path).stem
    
    # 保存中间结果 JSON（middle.json）
    middle_json_path = os.path.join(output_dir, f"{image_name}_middle.json")
    try:
        # middle_json 已经是 JSON 字符串，直接保存
        with open(middle_json_path, 'w', encoding='utf-8') as f:
            f.write(middle_json)
        print(f"中间结果 JSON 已保存到: {middle_json_path}")
    except Exception as e:
        print(f"警告: 保存中间结果 JSON 失败: {e}")
    
    # 保存解析结果 JSON
    json_path = os.path.join(output_dir, f"{image_name}_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'input_path': image_path,
            'cells': cells,
            'total_cells': len(cells)
        }, f, indent=4, ensure_ascii=False)
    print(f"解析结果 JSON 已保存到: {json_path}")
    
    # 绘制标注框
    annotated_path = os.path.join(output_dir, f"{image_name}_annotated.jpg")
    annotated_image, _ = draw_bboxes_from_cells(cells, image_path, annotated_path)
    
    return {
        'cells': cells,
        'json_path': json_path,
        'middle_json_path': middle_json_path,
        'annotated_path': annotated_path
    }


if __name__ == "__main__":
    import sys
    
    # 默认配置
    lang = "ch"
    output_dir = "mineru_output"
    
    # 默认处理 images 目录中的所有图片
    images_dir = "bookpic"
    
    # 解析命令行参数
    args_list = sys.argv[1:]
    i = 0
    backend = "vlm-http-client"
    server_url = "http://localhost:30000"
    number_arg = None
    type_arg = "all"  # 默认处理所有图片
    
    while i < len(args_list):
        if args_list[i] == "--lang" and i + 1 < len(args_list):
            lang = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--output-dir" and i + 1 < len(args_list):
            output_dir = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--backend" and i + 1 < len(args_list):
            backend = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--server-url" and i + 1 < len(args_list):
            server_url = args_list[i + 1]
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
            parse_image_with_annotation(
                input_path, 
                lang=lang,
                backend=backend,
                server_url=server_url,
                output_dir=output_dir
            )
        elif os.path.isdir(input_path):
            # 目录
            images_dir = input_path
        else:
            print(f"错误: {input_path} 不是有效的文件或目录")
            sys.exit(1)
    
    # 处理 images 目录中的所有图片
    if os.path.isdir(images_dir):
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.pdf')
        
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
        print(f"语言设置: {lang}")
        print(f"输出目录: {output_dir}")
        print(f"后端引擎: {backend}")
        print(f"服务器地址: {server_url}")
        
        # 创建 parser 实例（只初始化一次，复用）
        print("\n正在初始化 MinerU 解析器（仅初始化一次，后续图片将复用）...")
        parser = MinerUImageParser(lang=lang, backend=backend, server_url=server_url)
        print("解析器初始化完成，开始处理图片...\n")
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                image_file = os.path.relpath(image_path, images_dir)
            except ValueError:
                image_file = image_path
            print(f"\n[{idx}/{len(image_files)}] 处理: {image_file}")
            try:
                # 传入 parser 实例，避免重复初始化
                parse_image_with_annotation(
                    image_path, 
                    parser=parser,  # 复用同一个 parser 实例
                    backend=backend,
                    server_url=server_url,
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

