# load Aria model & tokenizer with vllm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
from IPython.display import Markdown, display
import requests
from transformers import AriaForConditionalGeneration, AutoProcessor
import requests
import torch
from PIL import Image
import json
from transformers import AutoTokenizer
import decord
from decord import VideoReader
from tqdm import tqdm
from typing import List
Image.MAX_IMAGE_PIXELS = None
import debugpy
from decord import cpu

# 启动debugpy并监听指定地址和端口
debugpy.listen(('127.0.0.1', 61347))

# 输出启动成功信息，告诉用户可以开始连接
print("Debugpy server is listening on 127.0.0.1:61347. Please connect your debugger.")

# 等待VS Code或其他调试客户端的连接
debugpy.wait_for_client()

# 在调试客户端连接后继续执行后面的代码
print("Debugger is connected, starting inference...")

def load_video(video_file, fps=1, time_range=None): #

    decord.bridge.set_bridge("torch")

    vr = VideoReader(video_file, ctx=cpu(0),num_threads=1)
    duration = len(vr)
    video_fps = vr.get_avg_fps()
    if time_range is not None:
        # 计算起始时间和截止时间对应的帧位置
        start_time = time_range[0]
        end_time = time_range[1]
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        num_frames = (end_frame-start_frame)*fps
        fps = round(video_fps/fps)
        frame_indices = [i for i in range(start_frame, end_frame, fps)]
        # video_time = end_time - start_time
        # frame_time = [i/video_fps for i in frame_idx]
    else:
        frame_indices = [int(duration / num_frames) * i for i in range(num_frames)]
    frames = vr.get_batch(frame_indices).numpy()
    frame_timestamps =  [i/video_fps for i in frame_indices]
    return [Image.fromarray(fr).convert("RGB") for fr in frames], frame_timestamps

def create_image_gallery(images, columns=3, spacing=20, bg_color=(200, 200, 200)):
    """
    Combine multiple images into a single larger image in a grid format.
    
    Parameters:
        image_paths (list of str): List of file paths to the images to display.
        columns (int): Number of columns in the gallery.
        spacing (int): Space (in pixels) between the images in the gallery.
        bg_color (tuple): Background color of the gallery (R, G, B).
    
    Returns:
        PIL.Image: A single combined image.
    """
    # Open all images and get their sizes
    img_width, img_height = images[0].size  # Assuming all images are of the same size

    # Calculate rows needed for the gallery
    rows = (len(images) + columns - 1) // columns

    # Calculate the size of the final gallery image
    gallery_width = columns * img_width + (columns - 1) * spacing
    gallery_height = rows * img_height + (rows - 1) * spacing

    # Create a new image with the calculated size and background color
    gallery_image = Image.new('RGB', (gallery_width, gallery_height), bg_color)

    # Paste each image into the gallery
    for index, img in enumerate(images):
        row = index // columns
        col = index % columns

        x = col * (img_width + spacing)
        y = row * (img_height + spacing)

        gallery_image.paste(img, (x, y))

    return gallery_image


def get_placeholders_for_videos(frames: List, timestamps=[]):
    contents = []
    if not timestamps:
        for i, _ in enumerate(frames):
            contents.append({"text": None, "type": "image"})
        contents.append({"text": "\n", "type": "text"})
    else:
        for i, (_, ts) in enumerate(zip(frames, timestamps)):
            contents.extend(
                [
                    {"text": f"[{int(ts)//60:02d}:{int(ts)%60:02d}]", "type": "text"},
                    {"text": None, "type": "image"},
                    {"text": "\n", "type": "text"}
                ]
            )
    return contents

import re
def convert_time_to_seconds(text):
    # 匹配未被引号包裹、带冒号的时间字段，如： "start_time": 0:49
    pattern_start = r'("start_time"\s*:\s*)(\d+:\d+)'
    pattern_end = r'("end_time"\s*:\s*)(\d+:\d+)'

    def replacer(match):
        prefix = match.group(1)  # "start_time":
        time_str = match.group(2)  # 0:49

        # 转换成秒
        minutes, seconds = map(int, time_str.split(":"))
        total_seconds = minutes * 60 + seconds
        return f'{prefix}{total_seconds:.1f}'  # 替换为字符串秒数形式
    text = re.sub(pattern_start, replacer, text)
    text = re.sub(pattern_end, replacer, text)
    
    return text
def escape_unescaped_quotes_in_reasoning(json_like_str):
    # 匹配 "reasoning": 后跟任意非大括号的内容，直到遇到 }
    def replacer(match):
        reasoning_text = match.group(1)
        # 替换不是 \" 的 "，使用负向前瞻与回顾确保不是已转义的引号
        escaped = re.sub(r'(?<!\\)"', r'\"', reasoning_text)
        return f'"reasoning": "{escaped}."'
    new_text = re.sub(r'"reasoning":\s*"(.+?)\."', replacer, json_like_str)
    return new_text
    # matches = re.findall(r'"reasoning":\s*"(.+?)\."\s*', json_like_str)


if __name__ == "__main__":
    model_id_or_path = "rhymes-ai/Aria"
    model = AriaForConditionalGeneration.from_pretrained(
        model_id_or_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)

    json_name = 'qa.json'
    cut_video_json_path = './Daily-Omni/' + json_name
    with open(cut_video_json_path, 'r', encoding='utf-8') as file:
        cut_video_json = json.load(file)

    audio_json_name = 'qa.json_r1-avqa_audio_des_split5_1024_time.json'
    audio_des_path = './Daily-Omni/result/' + audio_json_name
    with open(audio_des_path, 'r', encoding='utf-8') as file:
        audio_split5_json = json.load(file)
    with open("./Daliy-Omni/result/qa.json_r1-avqa_audio_subtitle_split5_1024.json", 'r', encoding='utf-8') as file:
        subtitle_split5_json = json.load(file)
        
    video_dir = "./Daily-Omni/Videos"
    mname = 'Aria'
    step_name = "video_time_range"
    result_dir = f'./Daily-Omni/Agent_result/memory'
   

    for index in range(len(cut_video_json)):
        item = cut_video_json[index]
        aitem = audio_split5_json[index]
        aitem_subtitle = subtitle_split5_json[index]

        vname = item["video_id"]
        result_path = os.path.join(result_dir,vname)
        os.makedirs(result_path, exist_ok=True) 
        result_path = os.path.join(result_path,f"{vname}_{mname}_{step_name}.json")
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as file:
                results = json.load(file)
            if item['Question'] in results and index>90:
                print("continue")
                continue
        else:
            results = {}
        
        video_path = os.path.join(video_dir, item["video_id"], item["video_id"]+"_video.mp4")
        video_duration = int(item["video_duration"][:-1])
        all_time = aitem["all_time"]

        videos = []
        content = []
        for tri in range(len(all_time)):
            time_range = all_time[tri]
            # Video
            frames, frame_timestamps = load_video(video_path, fps=1, time_range=time_range)#num_frames=128
            contents = get_placeholders_for_videos(frames, frame_timestamps)
            videos.extend(frames)
            ades = aitem['r1-avqa_audio_des_split5_1024_time'][tri]
            sdes = aitem_subtitle['r1-avqa_audio_subtitle_split5_1024'][tri]

            video_time = time_range[1] - time_range[0]
            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are uniformly sampled from it. "\
                f"This video is a {time_range[0]}-{time_range[1]}s segment extracted from a longer video." \
                f"Subtitle: {sdes}\n"\
                f"Audio description: {ades}\n"
            content.extend(contents)
            content.append({"text": time_instruciton, "type": "text"})

        question = f"Please carefully read the questions related to audio-visual perception, understanding, and reasoning abilities, closely observe the video frames and corresponding subtitles. "\
        f"Please determine which time segment(s) of the video provide the necessary information to answer the question and provide the corresponding reasoning. "\
        f"Question: {item['Question']}\n"\
        f"No need to answer the question itself, just identify the time range(s) and provide the corresponding reasoning.\n"\
        "Reply me with a structured output in JSON format:\n"\
        "{\"time_segments\": [{ \"start_time\": , \"end_time\": , \"reasoning\": }...]"
        content.append({"type": "text", "text": question})
        
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=videos, return_tensors="pt", max_image_size=490)
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model.generate(
                **inputs,
                max_new_tokens=4096,
                stop_strings=["<|im_end|>"],
                tokenizer=processor.tokenizer,
                do_sample=True,
                temperature=0.1
            )
            output_ids = output[0][inputs["input_ids"].shape[1]:]
            output_text = processor.decode(output_ids, skip_special_tokens=True)
        output_text = output_text.strip().replace("```json", "").replace("```<|im_end|>", "") #.replace("<|im_end|>", "") #
        # 解析为字典
        cleaned_data = escape_unescaped_quotes_in_reasoning(output_text)
        cleaned_data = convert_time_to_seconds(cleaned_data)
        data = json.loads(cleaned_data)
        print(data)
        results[item['Question']] = data    
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
