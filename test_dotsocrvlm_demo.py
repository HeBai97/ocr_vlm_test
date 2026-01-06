import re
import copy
import json
import pathlib
import random
from typing import List, Optional

from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import fitz
from PIL import Image, ImageDraw, ImageFont

from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image
from dots_ocr.utils.doc_utils import fitz_doc_to_image
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, pre_process_bboxes


class DotsOCRParser:
    """
    parse image or pdf file - 使用本地VLM推理
    """
    
    def __init__(self, 
            ip="127.0.0.1",
            port=8000,
            model_path="./weights/DotsOCR",  # 本地模型路径（use_hf=True时使用）
            model_name="dotsocr-model",  # vllm服务使用的模型名称（use_hf=False时使用），默认对应docker容器中的模型名称
            protocol="http",  # vllm服务协议
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=16384,
            num_thread=1,  # 本地推理时建议使用单线程，vllm推理时可以使用多线程
            dpi = 72, 
            min_pixels=None,
            max_pixels=None,
            use_hf=True,  # True: 使用HuggingFace本地推理, False: 使用vllm服务推理
            device=None,  # 指定GPU设备，如 "cuda:0", "cuda:1" 等，None表示自动选择（仅use_hf=True时有效）
            max_image_pixels=None,  # 限制图像最大像素数以减少内存使用，默认使用 MAX_PIXELS
        ):
        self.ip = ip
        self.port = port
        self.protocol = protocol
        self.dpi = dpi
        self.model_path = model_path
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.num_thread = num_thread
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.use_hf = use_hf
        self.device = device
        # 如果未指定 max_image_pixels，使用较小的值以减少内存占用（约 1.5M 像素，对应约 1225x1225）
        # 这比默认的 MAX_PIXELS (11M) 小很多，可以显著减少内存使用
        # 如果仍然遇到内存问题，可以进一步降低这个值，例如设置为 1000000 (约 1000x1000)
        self.max_image_pixels = max_image_pixels if max_image_pixels is not None else min(MAX_PIXELS, 1500000)

        if self.use_hf:
            self._load_hf_model()
            print(f"使用本地VLM模型推理，模型路径: {model_path}")
            print(f"num_thread 设置为 {self.num_thread} (本地推理建议使用单线程)")
        else:
            print(f"使用vllm服务推理，服务地址: {protocol}://{ip}:{port}")
            print(f"模型名称: {model_name}")
            print(f"num_thread 设置为 {self.num_thread} (vllm推理可以使用多线程)")
        
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    def _load_hf_model(self):
        """加载本地HuggingFace模型"""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from qwen_vl_utils import process_vision_info

        print(f"正在加载模型: {self.model_path}")
        
        # 确定使用的设备
        if self.device is None:
            # 自动选择空闲的GPU
            if torch.cuda.is_available():
                # 检查每个GPU的可用内存（使用 torch.cuda.mem_get_info）
                best_device = None
                max_free_memory = 0
                for i in range(torch.cuda.device_count()):
                    try:
                        free_memory, total_memory = torch.cuda.mem_get_info(i)
                        if free_memory > max_free_memory:
                            max_free_memory = free_memory
                            best_device = f"cuda:{i}"
                    except Exception:
                        continue
                if best_device:
                    self.device = best_device
                    print(f"自动选择设备: {self.device} (可用内存: {max_free_memory / 1024**3:.2f} GB)")
                else:
                    self.device = "cuda:0"
                    print(f"使用默认设备: {self.device}")
            else:
                self.device = "cpu"
                print(f"CUDA不可用，使用CPU")
        else:
            print(f"使用指定设备: {self.device}")
        
        # 尝试使用 flash_attention_2，如果不可用则使用默认的 attention
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True
        }
        
        # 如果指定了设备，使用 device_map 指定设备，否则使用 auto
        if self.device.startswith("cuda"):
            device_id = int(self.device.split(":")[1])
            model_kwargs["device_map"] = {"": device_id}
        else:
            model_kwargs["device_map"] = "auto"
        
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("使用 flash_attention_2")
        except ImportError:
            print("flash-attn 未安装，使用默认 attention")
        
        # 清理GPU缓存并设置内存分配策略
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 设置内存分配策略以减少碎片
            import os
            if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 使用低内存模式加载模型
        model_kwargs['low_cpu_mem_usage'] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs
        )
        
        # 设置为评估模式并禁用梯度计算
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
            use_fast=True
        )
        self.process_vision_info = process_vision_info
        print("模型加载完成！")

    def _inference_with_hf(self, image, prompt):
        """使用本地HuggingFace模型进行推理"""
        import torch
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 准备推理
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 将输入移动到模型所在的设备
        if hasattr(self.model, 'device'):
            device = self.model.device
        elif hasattr(self.model, 'hf_device_map'):
            # 如果使用 device_map，获取第一个设备的设备
            device = list(self.model.hf_device_map.values())[0] if self.model.hf_device_map else "cpu"
        else:
            device = self.device if self.device else "cpu"
        inputs = inputs.to(device)

        # 推理：生成输出（使用 torch.no_grad() 减少内存占用）
        # 进一步降低 max_new_tokens 以减少内存使用
        max_tokens = min(self.max_completion_tokens or 16384, 8192)  # 限制最大token数
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=False,  # 明确设置为 False 以避免采样相关的内存开销
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # 清理中间变量和 GPU 缓存
        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response

    def _inference_with_vllm(self, image, prompt):
        """使用vllm服务进行推理"""
        from dots_ocr.model.inference import inference_with_vllm
        
        response = inference_with_vllm(
            image,
            prompt, 
            model_name=self.model_name,
            protocol=self.protocol,
            ip=self.ip,
            port=self.port,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        return prompt

    def _parse_single_image(
        self, 
        origin_image, 
        prompt_mode, 
        source="image", 
        page_idx=0, 
        bbox=None,
        fitz_preprocess=False,
        ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None: 
            assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: 
            assert max_pixels <= MAX_PIXELS, f"max_pixels should <= {MAX_PIXELS}"

        # 预处理图片（使用限制后的 max_pixels 以减少内存占用）
        effective_max_pixels = max_pixels if max_pixels is not None else self.max_image_pixels
        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels or MIN_PIXELS, max_pixels=effective_max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels or MIN_PIXELS, max_pixels=effective_max_pixels)
        
        input_height, input_width = image.height, image.width
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)

        # 根据配置选择推理方式
        if self.use_hf:
            response = self._inference_with_hf(image, prompt)
        else:
            response = self._inference_with_vllm(image, prompt)
        
        result = {
            'page_no': page_idx,
            "input_height": input_height,
            "input_width": input_width
        }
        
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            cells, filtered = post_process_output(
                response, 
                prompt_mode, 
                origin_image, 
                image,
                min_pixels=min_pixels, 
                max_pixels=max_pixels,
            )
            return {'cells': cells, 'page_no': page_idx}

        return {'cells': response, 'page_no': page_idx}
    
    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(
            origin_image, 
            prompt_mode, 
            source="image", 
            bbox=bbox, 
            fitz_preprocess=fitz_preprocess
        )
        result['file_path'] = input_path
        return [result]
    
    
    def load_images_from_pdf_bytes(self, pdf_bytes: bytes, start_page_id: int = 0, end_page_id: Optional[int] = None) -> list:
        images = []
        start_page_id = start_page_id
        end_page_id = end_page_id
        with fitz.open("temp.pdf", stream=pdf_bytes) as doc:
            pdf_page_num = doc.page_count
            end_page_id = (
                end_page_id
                if end_page_id is not None and end_page_id >= 0
                else pdf_page_num - 1
            )
            if end_page_id > pdf_page_num - 1:
                print('end_page_id is out of range, use images length')
                end_page_id = pdf_page_num - 1
            
            for index in range(start_page_id, end_page_id + 1):
                if start_page_id <= index <= end_page_id:
                    page = doc[index]
                    img = fitz_doc_to_image(page, target_dpi=self.dpi)
                    images.append(img)
        
        return images
    
    def parse_pdf(self, pdf_bytes: bytes, prompt_mode: str, start_page_id: int = 0, end_page_id_exclusive: Optional[int] = None):
        images_origin = self.load_images_from_pdf_bytes(
            pdf_bytes,
            start_page_id=start_page_id,
            end_page_id=end_page_id_exclusive
        )
        total_pages = len(images_origin)
        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "source": "pdf",
                "page_idx": i + max(0, int(start_page_id or 0)),
            } for i, image in enumerate(images_origin)
        ]
        
        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        # 本地VLM推理时，建议使用单线程或少量线程
        num_thread = min(total_pages, self.num_thread)
        start_log = max(0, int(start_page_id or 0))
        end_log = start_log + total_pages
        print(f"解析PDF页面 [{start_log}-{end_log})，共 {total_pages} 页，使用 {num_thread} 个线程...")

        results = []
        with ThreadPool(num_thread) as pool:
            with tqdm(total=total_pages, desc="处理PDF页面") as pbar:
                for result in pool.imap_unordered(_execute_task, tasks):
                    results.append(result)
                    pbar.update(1)
                    
        results.sort(key=lambda x: x['page_no'])
        
        return results
    


def _is_latin_start(text: str) -> bool:
    try:
        if len(text) == 0:
            return False
        return text[0].isalpha()
    except Exception as e:
        print(text)
        raise e

def _is_end_with_end_symbol(text: str) -> bool:
    try:
        if len(text) == 0:
            return False
        return text[-1] in ['.', '。', '!', '！', '?', '？']
    except Exception as e:
        print(text)
        raise e

def _add_position_to_doc(doc: dict, page_num: int, bbox: List[int]):
    if 'page_num_int' not in doc:
        doc['page_num_int'] = []
        doc['position_int'] = []
        doc['top_int'] = []
    doc['page_num_int'].append(page_num)
    doc['position_int'].append([page_num, bbox[0], bbox[2], bbox[1], bbox[3]])
    doc['top_int'].append(bbox[1])
    
def _merge_all_cells_on_block(block: List[dict], tag: str = 'text') -> str:
    lines = block['lines']
    res = ''
    for line in lines:
        for span in line['spans']:
            cur_text = span[tag]
            if _is_latin_start(cur_text):
                if len(res) > 0 and res[-1] == '-':
                    res = res[:-1] + cur_text
                else:
                    res += ' ' + cur_text
            else:
                res += cur_text
    return res.lstrip()

def find_caption_on_nearby_cells(cells: List[dict], idx: int) -> Optional[dict]:
    if idx - 1 >= 0 and cells[idx - 1]['category'] == 'Caption':
        return cells[idx - 1]
    if idx + 1 < len(cells) and cells[idx + 1]['category'] == 'Caption':
        return cells[idx + 1]
    return None

def _bbox_vertical_gap(bbox_a: List[int], bbox_b: List[int]) -> int:
    """Return non-negative vertical gap between two boxes; 0 if overlapping vertically; negative means overlapping (clamped to 0)."""
    try:
        top_a, bottom_a = int(bbox_a[1]), int(bbox_a[3])
        top_b, bottom_b = int(bbox_b[1]), int(bbox_b[3])
    except Exception:
        return 999999
    # a above b
    if bottom_a <= top_b:
        return top_b - bottom_a
    # b above a
    if bottom_b <= top_a:
        return top_a - bottom_b
    # vertical overlap
    return 0


def _bbox_horizontal_overlap_ratio(bbox_a: List[int], bbox_b: List[int]) -> float:
    """Compute horizontal overlap ratio with respect to the smaller width of the two boxes."""
    try:
        left_a, right_a = int(bbox_a[0]), int(bbox_a[2])
        left_b, right_b = int(bbox_b[0]), int(bbox_b[2])
    except Exception:
        return 0.0
    inter_left = max(left_a, left_b)
    inter_right = min(right_a, right_b)
    intersection = max(0, inter_right - inter_left)
    width_a = max(1, right_a - left_a)
    width_b = max(1, right_b - left_b)
    base = min(width_a, width_b)
    return intersection / base if base > 0 else 0.0

def find_table_caption_on_nearby_cells(cells: List[dict], table_index: int,
                                       max_scan_neighbors: int = 5,
                                       max_vertical_gap: int = 160,
                                       min_horizontal_overlap_ratio: float = 0.2) -> Optional[dict]:
    """
    查找与给定 Table 单元相邻的 Caption 单元。

    策略：
    - 在 `table_index` 前后最多 `max_scan_neighbors` 个元素内查找 `category == "Caption"` 的单元。
    - 仅接受与 Table 垂直相邻且间距不超过 `max_vertical_gap` 的候选。
    - 要求候选与 Table 在水平方向有一定重叠（默认重叠比例>= `min_horizontal_overlap_ratio`）。
    - 多个候选时，优先垂直间距更小者；若并列，再取水平重叠比例更大者。

    返回：最匹配的 caption 单元字典；若无匹配返回 None。
    """
    if not (0 <= table_index < len(cells)):
        return None
    table_cell = cells[table_index]
    if table_cell.get("category") != "Table":
        return None
    table_bbox = [int(x) for x in table_cell.get("bbox", [])] or None
    if not table_bbox or len(table_bbox) < 4:
        return None

    best = None
    best_key = (999999, -1.0)  # (vertical_gap, -horizontal_overlap_ratio)

    start = max(0, table_index - max_scan_neighbors)
    end = min(len(cells) - 1, table_index + max_scan_neighbors)

    for i in range(start, end + 1):
        if i == table_index:
            continue
        c = cells[i]
        if c.get("category") != "Caption":
            continue
        bbox = c.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue
        bbox = [int(x) for x in bbox]
        vgap = _bbox_vertical_gap(table_bbox, bbox)
        if vgap > max_vertical_gap:
            continue
        hoverlap = _bbox_horizontal_overlap_ratio(table_bbox, bbox)
        if hoverlap < min_horizontal_overlap_ratio:
            continue
        rank_key = (vgap, -hoverlap)
        if rank_key < best_key:
            best_key = rank_key
            best = c

    return best

def draw_bboxes_from_cells(cells: List[dict], image_path: str, output_path: str = None):
    """
    从DotsOCR解析的cells结果，在图片上绘制标注框
    
    Args:
        cells: DotsOCR解析返回的cells列表
        image_path: 输入图片路径
        output_path: 输出图片路径，如果为None则自动生成
    """
    # 打开原始图片
    image = Image.open(image_path).convert("RGB")
    
    # 定义颜色映射（不同类别使用不同颜色）
    category_colors = {
        'Text': (255, 0, 0),           # 红色
        'Title': (0, 255, 0),           # 绿色
        'List-item': (0, 0, 255),       # 蓝色
        'Picture': (255, 255, 0),       # 黄色
        'Table': (255, 0, 255),         # 紫色
        'Formula': (0, 255, 255),       # 青色
        'Caption': (255, 165, 0),       # 橙色
        'Page-footer': (128, 128, 128), # 灰色
        'Page-header': (128, 128, 128), # 灰色
        'Footnote': (128, 128, 128),    # 灰色
    }
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            font = ImageFont.load_default()
    
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
            
        category = cell.get('category', 'Text')
        text = cell.get('text', '').strip()
        bbox = cell.get('bbox', [])
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # 获取颜色
            color = category_colors.get(category, (128, 128, 128))
            
            # 绘制半透明填充矩形
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=color + (50,))
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # 绘制边框（更粗一些以便清晰可见）
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签文本（在bbox左上角）
            label_text = f"{idx+1}:{category}"
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

def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = t
    d["content_sm_ltks"] = t


def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="ch", callback=None, model_path="./weights/DotsOCR", device=None, 
          use_hf=True, ip="127.0.0.1", port=8000, model_name="dotsocr-model", 
          protocol="http", num_thread=None, **kwargs):
    """
    使用VLM模型进行PDF解析
    
    Args:
        filename: PDF文件路径
        binary: PDF二进制数据（可选）
        from_page: 起始页码
        to_page: 结束页码
        lang: 语言（默认中文）
        callback: 进度回调函数
        model_path: 本地VLM模型路径（use_hf=True时使用）
        device: GPU设备，如 "cuda:0", "cuda:1" 等，None表示自动选择（仅use_hf=True时有效）
        use_hf: True使用本地HuggingFace推理，False使用vllm服务推理
        ip: vllm服务IP地址（use_hf=False时使用）
        port: vllm服务端口（use_hf=False时使用）
        model_name: vllm服务使用的模型名称（use_hf=False时使用）
        protocol: vllm服务协议（use_hf=False时使用）
        num_thread: 线程数，None时自动设置（use_hf=True时为1，use_hf=False时为64）
    """
    
    if not lang:
        lang = 'ch'
    is_english = lang.lower() == "english"

    doc = {
        "docnm_kwd": filename,
    }

    # 根据配置选择推理方式
    if num_thread is None:
        num_thread = 1 if use_hf else 64
    
    parser = DotsOCRParser(
        model_path=model_path, 
        model_name=model_name,
        use_hf=use_hf, 
        num_thread=num_thread, 
        device=device,
        ip=ip,
        port=port,
        protocol=protocol
    )
    callback(0.2, "开始解析...")
    binary = open(filename, 'rb').read() if binary is None else binary
    callback(0.3, "解析中...")
    
    # 处理页面范围
    start_page_id = max(0, int(from_page or 0))
    end_page_id_exclusive = None if to_page is None or int(to_page) < 0 else int(to_page)
    pdf_results = parser.parse_pdf(
        binary, 
        "prompt_layout_all_en", 
        start_page_id=start_page_id, 
        end_page_id_exclusive=end_page_id_exclusive
    )
    res = []

    # === 按位置排序确保正确顺序 ===
    def get_element_position(cell):
        """根据元素在页面中的位置排序（从上到下，从左到右）"""
        bbox = cell.get('bbox', [0, 0, 0, 0])
        return (-bbox[3], bbox[0])

    # === 维护已使用的cell集合，避免重复引用 ===
    used_indices = set()  # 存储 (page_idx, cell_idx) 元组

    def log_chunk_event(event, page_num, bbox, text, extra=None):
        preview = (text or "").replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        extrastr = f" | {extra}" if extra else ""
        print(f"[chunk-{event}] page={page_num} bbox={bbox} text='{preview}'{extrastr}")

    for page_idx, page in enumerate(pdf_results):
        cells = page["cells"]

        normalized_cells = []
        for idx, cell in enumerate(cells):
            if isinstance(cell, dict):
                normalized_cells.append((idx, cell))
                continue
            parsed = None
            if isinstance(cell, str):
                try:
                    parsed = json.loads(cell)
                except Exception:
                    pass
            if isinstance(parsed, dict):
                normalized_cells.append((idx, parsed))
                continue
            print(f"[chunk-warn] invalid cell skipped index={idx}, type={type(cell)}")

        page_num = int(page['page_no']) + 1

        # === 修复：按视觉位置排序cells ===
        sorted_cells = sorted(
            normalized_cells,
            key=lambda x: get_element_position(x[1])
        )

        # === 参考mineru逻辑：维护文本合并状态（按页重置） ===
        prev_text_doc = None
        prev_text_bbox = None
        is_prev_block_non_text = False

        # === 按排序后的顺序处理cells ===
        for idx, cell in sorted_cells:
            cell_type = cell['category']
            cell_bbox = [int(x) for x in cell['bbox']]
            full_cell_id = (page_idx, idx)

            # 跳过已被使用的 cell（比如被作为 caption 被引用过）
            if full_cell_id in used_indices:
                continue

            # 跳过非正文元素
            if cell_type in ['Page-header', 'Page-footer', 'Footnote']:
                continue

            # === Caption 自身不单独输出，只能被引用 ===
            if cell_type == 'Caption':
                continue

            _d = copy.deepcopy(doc)
            text = cell.get('text', '').strip()

            # === 处理Text和List-item类型：参考mineru的文本合并逻辑 ===
            if cell_type in ['Text', 'List-item']:
                # 智能合并：如果前一个是文本且当前文本看起来是延续
                if prev_text_doc is not None and not is_prev_block_non_text:
                    prev_text = prev_text_doc['content_with_weight']

                    merge_ok = True
                    vgap = None
                    hover = None
                    if prev_text_bbox is not None:
                        vgap = _bbox_vertical_gap(prev_text_bbox, cell_bbox)
                        hover = _bbox_horizontal_overlap_ratio(prev_text_bbox, cell_bbox)
                        if vgap > 60 or hover < 0.3:
                            merge_ok = False

                    # 判断是否应该合并：前文未结束或当前文本以小写开头
                    if merge_ok and (not _is_end_with_end_symbol(prev_text) or (text and text[0].islower())):
                        # 合并到前一个文档
                        merged_text = prev_text + ' ' + text
                        _add_position_to_doc(prev_text_doc, page_num, cell_bbox)
                        tokenize(prev_text_doc, merged_text, is_english)
                        is_prev_block_non_text = False
                        prev_text_bbox = cell_bbox
                        log_chunk_event("merge", page_num, cell_bbox, text, extra=f"vgap={vgap}, hover={hover}")
                        continue

                # 创建新的文本块
                _add_position_to_doc(_d, page_num, cell_bbox)
                tokenize(_d, text, is_english)
                prev_text_doc = _d
                prev_text_bbox = cell_bbox
                is_prev_block_non_text = False
                res.append(_d)
                log_chunk_event("new", page_num, cell_bbox, text)

            # === 处理Picture：优先使用Caption ===
            elif cell_type == "Picture":
                caption_cell = find_caption_on_nearby_cells(cells, idx)
                caption_text = ""
                
                if caption_cell is not None:
                    try:
                        caption_idx = cells.index(caption_cell)
                        caption_id = (page_idx, caption_idx)
                        if caption_id not in used_indices:
                            caption_text = caption_cell['text'].strip()
                            used_indices.add(caption_id)  # 标记Caption为已使用
                            # 同时添加Caption的位置信息
                            _add_position_to_doc(_d, page_num, [int(x) for x in caption_cell['bbox']])
                    except (ValueError, KeyError):
                        pass
                
                # 组合图片内容
                if caption_text:
                    text = f"{caption_text}"
                elif text:
                    text = f"{text}"
                else:
                    text = "[图片]"
                    
                _add_position_to_doc(_d, page_num, cell_bbox)
                tokenize(_d, text, is_english)
                prev_text_doc = None
                prev_text_bbox = None
                is_prev_block_non_text = True
                res.append(_d)
                log_chunk_event("picture", page_num, cell_bbox, text)

            # === 处理Table：优先使用Caption ===
            elif cell_type == "Table":
                caption_cell = find_table_caption_on_nearby_cells(cells, idx)
                caption_text = ""
                
                if caption_cell is not None:
                    try:
                        caption_idx = cells.index(caption_cell)
                        caption_id = (page_idx, caption_idx)
                        if caption_id not in used_indices:
                            caption_text = caption_cell['text'].strip()
                            used_indices.add(caption_id)  # 标记Caption为已使用
                            # 添加Caption的位置信息
                            _add_position_to_doc(_d, page_num, [int(x) for x in caption_cell['bbox']])
                    except (ValueError, KeyError):
                        pass
                
                # 组合表格内容
                table_parts = []
                if caption_text:
                    table_parts.append(f"{caption_text}")
                if text:
                    table_parts.append(f"{text}")
                
                if table_parts:
                    text = "\n".join(table_parts)
                else:
                    text = "[表格]"
                    
                _add_position_to_doc(_d, page_num, cell_bbox)
                tokenize(_d, text, is_english)
                prev_text_doc = None
                prev_text_bbox = None
                is_prev_block_non_text = True
                res.append(_d)
                log_chunk_event("table", page_num, cell_bbox, text)

            # === 处理其他类型 ===
            else:
                if text:
                    _add_position_to_doc(_d, page_num, cell_bbox)
                    tokenize(_d, text, is_english)
                    prev_text_doc = None
                    prev_text_bbox = None
                    is_prev_block_non_text = True
                    res.append(_d)
                    log_chunk_event("other", page_num, cell_bbox, text)
                else:
                    continue  # 跳过没有文本的未知类型

    callback(0.7, "解析完成！")
    return res


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


def parse_image_with_annotation(image_path: str, 
                                 parser: DotsOCRParser = None,
                                 model_path: str = "./weights/DotsOCR", 
                                 prompt_mode: str = "prompt_layout_all_en", 
                                 output_dir: str = "dotsocr_output", 
                                 device=None,
                                 use_hf=False,  # 默认使用 vllm (openai-http)
                                 ip="127.0.0.1", 
                                 port=8000, 
                                 model_name="dotsocr-model", 
                                 protocol="http"):
    """
    解析图片并生成标注图片
    
    Args:
        image_path: 输入图片路径
        parser: 可选的 DotsOCRParser 实例，如果提供则复用（避免重复初始化）
        model_path: 本地VLM模型路径（use_hf=True时使用）
        prompt_mode: 提示模式
        output_dir: 输出目录，默认为 "dotsocr_output"
        device: GPU设备，如 "cuda:0", "cuda:1" 等，None表示自动选择（仅use_hf=True时有效）
        use_hf: True使用本地HuggingFace推理，False使用vllm服务推理（默认False，使用openai-http）
        ip: vllm服务IP地址（use_hf=False时使用）
        port: vllm服务端口（use_hf=False时使用）
        model_name: vllm服务使用的模型名称（use_hf=False时使用）
        protocol: vllm服务协议（use_hf=False时使用）
    """
    import os
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有提供 parser，则创建一个新的
    if parser is None:
        parser = DotsOCRParser(
            model_path=model_path, 
            model_name=model_name,
            use_hf=use_hf, 
            num_thread=1, 
            device=device,
            ip=ip,
            port=port,
            protocol=protocol
        )
    
    print(f"解析图片: {image_path}")
    
    # 解析图片
    image_name = pathlib.Path(image_path).stem
    results = parser.parse_image(
        input_path=image_path,
        filename=image_name,
        prompt_mode=prompt_mode,
        save_dir=output_dir,
        fitz_preprocess=False
    )
    
    if not results:
        print("解析失败，未返回结果")
        return None
    
    result = results[0]
    cells = result.get('cells', [])
    
    print(f"解析完成，共找到 {len(cells)} 个元素")
    
    # 保存JSON结果
    json_path = os.path.join(output_dir, f"{image_name}_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'input_path': image_path,
            'cells': cells,
            'page_no': result.get('page_no', 0),
            'input_width': result.get('input_width', 0),
            'input_height': result.get('input_height', 0)
        }, f, indent=4, ensure_ascii=False)
    print(f"JSON结果已保存到: {json_path}")
    
    # 绘制标注框
    annotated_path = os.path.join(output_dir, f"{image_name}_annotated.jpg")
    annotated_image, _ = draw_bboxes_from_cells(cells, image_path, annotated_path)
    
    return {
        'cells': cells,
        'json_path': json_path,
        'annotated_path': annotated_path
    }


if __name__ == "__main__":
    import sys
    import os
    
    def dummy(prog=None, msg=""):
        print(f"[{prog}] {msg}")
    
    # 使用本地VLM模型路径
    model_path = "./weights/DotsOCR"  # 修改为你的模型路径
    
    # 默认配置
    use_hf = False  # 默认使用 vllm (openai-http)
    device_arg = None
    ip_arg = "127.0.0.1"
    port_arg = 8000
    model_name_arg = "dotsocr-model"  # 默认模型名称，对应docker容器中的模型名称
    output_dir = "dotsocr_output"
    
    # 默认处理 bookpic 目录中的所有图片
    images_dir = "bookpic"
    
    # 从环境变量或命令行参数获取设备
    device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device:
        # 如果设置了 CUDA_VISIBLE_DEVICES，使用 cuda:0（因为环境变量会重新映射设备）
        device = "cuda:0"
    else:
        device = None  # 自动选择
    
    # 解析命令行参数
    args_list = sys.argv[1:]
    i = 0
    number_arg = None
    type_arg = "all"  # 默认处理所有图片
    
    while i < len(args_list):
        if args_list[i] == "--device" and i + 1 < len(args_list):
            device_arg = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--use-hf":
            use_hf = True
            args_list = args_list[:i] + args_list[i+1:]
            continue
        elif args_list[i] == "--use-vllm":
            use_hf = False
            args_list = args_list[:i] + args_list[i+1:]
            continue
        elif args_list[i] == "--ip" and i + 1 < len(args_list):
            ip_arg = args_list[i + 1]
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--port" and i + 1 < len(args_list):
            port_arg = int(args_list[i + 1])
            args_list = args_list[:i] + args_list[i+2:]
            continue
        elif args_list[i] == "--model-name" and i + 1 < len(args_list):
            model_name_arg = args_list[i + 1]
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
    
    if device_arg:
        device = device_arg
    
    # 如果没有提供参数，默认处理 bookpic 目录
    if len(args_list) == 0:
        # 默认批量处理模式
        pass
    elif len(args_list) == 1 and args_list[0] == "--image":
        # 如果只有 --image 参数，使用默认目录
        args_list = []
    elif len(args_list) > 0:
        # 如果提供了命令行参数，使用指定的文件或目录
        input_path = args_list[0]
        if os.path.isfile(input_path):
            # 单个文件
            input_file = input_path
            args_list = args_list[1:]
            
            # 自动检测文件类型
            input_file_lower = input_file.lower()
            is_image_file = input_file_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'))
            is_pdf_file = input_file_lower.endswith('.pdf')
            
            # 判断是图片还是PDF
            if "--image" in args_list or (is_image_file and not is_pdf_file):
                # 图片解析模式
                parse_image_with_annotation(
                    input_file, 
                    model_path=model_path, 
                    device=device,
                    use_hf=use_hf,
                    ip=ip_arg,
                    port=port_arg,
                    model_name=model_name_arg,
                    output_dir=output_dir
                )
            else:
                # PDF解析模式
                from_page = int(args_list[1]) if len(args_list) > 1 else 0
                to_page = int(args_list[2]) if len(args_list) > 2 else 10
                
                if use_hf:
                    print(f"使用本地VLM模型: {model_path}")
                    if device:
                        print(f"使用设备: {device}")
                else:
                    print(f"使用vllm服务 (openai-http): {ip_arg}:{port_arg}")
                    print(f"模型名称: {model_name_arg}")
                
                print(f"解析文件: {input_file}, 页面范围: {from_page}-{to_page}")
                
                res = chunk(
                    input_file, 
                    from_page=from_page, 
                    to_page=to_page, 
                    callback=dummy,
                    model_path=model_path,
                    device=device,
                    use_hf=use_hf,
                    ip=ip_arg,
                    port=port_arg,
                    model_name=model_name_arg
                )
                
                output_file = os.path.join(output_dir, 'dotsocr_vlm.json')
                os.makedirs(output_dir, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(res, f, indent=4, ensure_ascii=False)
                
                print(f"\n解析结果已保存到: {output_file}")
                print(f"共解析 {len(res)} 个文本块")
            sys.exit(0)
        elif os.path.isdir(input_path):
            # 目录
            images_dir = input_path
            args_list = []
        else:
            print(f"错误: {input_path} 不是有效的文件或目录")
            sys.exit(1)
    
    # 批量处理 images_dir 目录中的所有图片
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
        if use_hf:
            print(f"使用本地VLM模型: {model_path}")
            if device:
                print(f"使用设备: {device}")
        else:
            print(f"使用vllm服务 (openai-http): {ip_arg}:{port_arg}")
            print(f"模型名称: {model_name_arg}")
        print(f"输出目录: {output_dir}")
        
        # 创建 parser 实例（只初始化一次，复用）
        print("\n正在初始化 DotsOCR 解析器（仅初始化一次，后续图片将复用）...")
        parser = DotsOCRParser(
            model_path=model_path, 
            model_name=model_name_arg,
            use_hf=use_hf, 
            num_thread=1, 
            device=device,
            ip=ip_arg,
            port=port_arg,
            protocol="http"
        )
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
                    model_path=model_path,
                    device=device,
                    use_hf=use_hf,
                    ip=ip_arg,
                    port=port_arg,
                    model_name=model_name_arg,
                    output_dir=output_dir
                )
            except Exception as e:
                print(f"处理 {image_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n所有图片处理完成！结果保存在: {output_dir}")
    else:
        print(f"错误: {images_dir} 目录不存在")
        print("\n用法:")
        print("  # 批量处理 bookpic 目录（默认）")
        print("  python test_dotsocrvlm_demo.py")
        print("  # 处理单个图片")
        print("  python test_dotsocrvlm_demo.py image.png")
        print("  # 处理指定目录")
        print("  python test_dotsocrvlm_demo.py /path/to/images/")
        print("  # PDF解析")
        print("  python test_dotsocrvlm_demo.py document.pdf 0 10")
        print("\n选项:")
        print("  --device cuda:0         指定GPU设备（仅use_hf=True时有效）")
        print("  --use-hf                使用本地HuggingFace推理")
        print("  --use-vllm              使用vllm服务推理（默认，openai-http）")
        print("  --ip IP                  vllm服务IP（默认: 127.0.0.1）")
        print("  --port PORT              vllm服务端口（默认: 8000）")
        print("  --model-name NAME       vllm模型名称（默认: dotsocr-model）")
        print("  --output-dir DIR         输出目录（默认: dotsocr_output）")
        print("  --number N               处理图片数量（仅 --type random 时有效）")
        print("  --type random/all       处理类型：random=随机抽取，all=处理所有（默认: all）")
        sys.exit(1)

