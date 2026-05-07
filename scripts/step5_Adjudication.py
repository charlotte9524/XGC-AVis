#/tos-bjml-researcheval/caoyuqin/env/onellm
# transformers==4.52.3
import os
import json
import time
max_retries = 5
import math
from google import genai
from google.genai import types
import copy
import io
from pydub import AudioSegment
import cv2
import moviepy.editor as mp
client = genai.Client(api_key='xx')


def closest_multiple_of_5(time_range,video_duration):
    start_time = math.floor(float(time_range[0]) / 5) * 5
    end_time = math.ceil(float(time_range[1]) / 5) * 5
    if end_time > video_duration:
        end_time = video_duration
    if start_time > video_duration:
        start_time = video_duration-5
    if start_time == end_time:
        start_time = end_time-5
    time_range = [start_time,end_time]
    return time_range

def merge(time_des, aitem_time): #, vdes,video_duration,all_time
    merge_reason = ""
    for i in range(len(time_des)):
        cur_reason = time_des[i]['reasoning']
        cur_reason = cur_reason.replace("\n","")
        merge_reason += f"{float(time_des[i]['start_time']):.2f}s–{float(time_des[i]['end_time']):.2f}s: {cur_reason}"
    # return merge_reason
    merge_time = []
    merge_time.append(closest_multiple_of_5([time_des[0]["start_time"],time_des[0]["end_time"]],video_duration))
    if len(time_des) > 1:
        for i in range(1, len(time_des)):
            if float(time_des[i]["start_time"])>=video_duration:
                continue
            cur_time = closest_multiple_of_5([time_des[i]["start_time"],time_des[i]["end_time"]],video_duration)
            if cur_time[0] < merge_time[-1][1]:
                merge_time[-1][1] = cur_time[1]
            else:
                merge_time.append(cur_time)
    
    cur_merge_time = []
    for i in range(len(merge_time)):
        for j in range(len(aitem_time)):
            if aitem_time[j][0] >= merge_time[i][0] and aitem_time[j][0] < merge_time[i][1]:
                cur_merge_time.append(aitem_time[j])
            elif aitem_time[j][1] > merge_time[i][0] and aitem_time[j][1] <= merge_time[i][1]:
                cur_merge_time.append(aitem_time[j])
            else:
                continue


    return cur_merge_time, merge_reason #, merge_ades

def frame_to_timecode(frame_idx, fps):
    total_seconds = frame_idx / fps
    m = int(total_seconds // 60)
    s = int(total_seconds % 60)
    return f"{m}:{s:02d}"
def extract_frames_and_audio(video_path, audio_path, extract_fps=1, time_range=None):
    # 打开视频
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps

    if time_range is None:
        start_time = 0
        end_time = total_duration
    else:
        start_time, end_time= time_range[0],time_range[1]
        if end_time is None or end_time > total_duration:
            end_time = total_duration
    start_frame_idx = int(start_time * fps)
    end_frame_idx = int(end_time * fps)
    if end_frame_idx > total_frames:
        end_frame_idx = total_frames

    duration_frames = end_frame_idx - start_frame_idx
    if duration_frames <= 0:
        raise ValueError("Invalid start_time and end_time combination.")

    frame_segmet = int(fps*extract_fps)
    # 如果时长不足，调整帧数
    key_frames, time_segments = [], []      # 存储 JPEG 编码的帧
    for i in range(total_frames):
        if i < start_frame_idx:
            continue
        if i >= end_frame_idx:
            break
        if i % frame_segmet == 0:
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if not ret:
                print(f"读取帧 {i} 失败")
                continue

            _, jpeg_frame = cv2.imencode('.jpg', frame)
            key_frames.append(jpeg_frame.tobytes())
            # i_timecode = frame_to_timecode(i, fps)
            i_timecode = int(i/fps)
            time_segments.append(i_timecode)

    video.release()

    # 提取指定时间段音频
    if not os.path.exists(audio_path):
        video_clip = mp.VideoFileClip(video_path).subclip(start_time, end_time)
        audio = video_clip.audio
        audio.write_audiofile(audio_path, verbose=False, logger=None)

    # 读取音频为字节
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    return key_frames, audio_bytes, time_segments

def find_subtitle(all_time,ades,sdes,vdess,start_time,end_time):
    cur_vdes = []
    for vi in range(len(vdess)):
        cur_vdes.append("")
    cur_ads = ""
    cur_sdes = ""
    for j in range(len(all_time)):
        if all_time[j][0] >= start_time and all_time[j][0] < end_time:
            for vi in range(len(vdess)):
                vdes = vdess[vi]
                cur_vdes[vi] += f"{all_time[j][0]}s-{all_time[j][1]}s: {vdes[j].replace('<|im_end|>','')} "
            cur_sdes += f"{all_time[j][0]}s-{all_time[j][1]}s: {sdes[j][0]} "
            cur_ads += f"{all_time[j][0]}s-{all_time[j][1]}s: {ades[j]} "
        elif all_time[j][1] > start_time and all_time[j][1] <= end_time:
            for vi in range(len(vdess)):
                vdes = vdess[vi]
                cur_vdes[vi] += f"{all_time[j][0]}s-{all_time[j][1]}s: {vdes[j].replace('<|im_end|>','')} "
            cur_sdes += f"{all_time[j][0]}s-{all_time[j][1]}s: {sdes[j][0]} "
            cur_ads += f"{all_time[j][0]}s-{all_time[j][1]}s: {ades[j]} "
        else:
            continue
    return cur_vdes, cur_sdes, cur_ads
if __name__ == "__main__":
    json_name = 'qa.json'
    cut_video_json_path = './Daliy-Omni/' + json_name
    with open(cut_video_json_path, 'r', encoding='utf-8') as file:
        cut_video_json = json.load(file)

    video_des_json_name=[
        ['Aria_video_time_range','time_segments', 'Aria_video_des_split5', 'video_des_split5'],
        ['Qwen2.5-Omni_video_time_range','', 'Qwen2.5-Omni_video_des_split5', 'video_des_split5']
    ]
    prev_model_name = "gemini-2.0-flash"
    with open(f'./Daliy-Omni/Agent_result/Qwen2.5-Omni/Aria_video_time_range_{prev_model_name}+subtitle.json', 'r', encoding='utf-8') as file:
        aria_response = json.load(file)
    with open(f'./Daliy-Omni/Agent_result/Qwen2.5-Omni/Qwen2.5-Omni_video_time_range_{prev_model_name}+subtitle.json', 'r', encoding='utf-8') as file:
        qwen_response = json.load(file)
    time_dir = f'./Daliy-Omni/Agent_result/memory'

    audio_json_name = 'qa.json_r1-avqa_audio_des_split5_1024_time.json'
    audio_des_path = './Daliy-Omni/result/' + audio_json_name
    with open(audio_des_path, 'r', encoding='utf-8') as file:
        audio_des_split5 = json.load(file)
    audio_subtitle_path = './Daliy-Omni/result/qa.json_r1-avqa_audio_subtitle_split5_1024.json'
    with open(audio_subtitle_path, 'r', encoding='utf-8') as file:
        audio_subtitle_split5 = json.load(file)
    
    model_name = "gemini-2.0-flash_select"
    abc = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    video_dir = './Daliy-Omni/Videos'
    result_dir = './Daliy-Omni/Agent_result/Qwen2.5-Omni'

    result_path = f'{result_dir}/Aria_Qwen2.5-Omni_timerange_subtitle_withoutvdes_{model_name}.json'
    results = []
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as file:
            results = json.load(file)
                
    for index in range(len(results), len(cut_video_json)):#
        item = cut_video_json[index]
        aitem_des = audio_des_split5[index]
        aitem_subtitle = audio_subtitle_split5[index]
        aitem_time = aitem_des["all_time"]
        video_duration = int(item["video_duration"][:-1])
        question = item["Question"]
        vname = item['video_id']
        video_path = os.path.join(video_dir, vname, vname+"_video.mp4")
        sound_path = os.path.join(video_dir, vname, vname+"_audio.wav")
        
        response_list = []
        response_list.append(aria_response[index][prev_model_name][0])
        response_list.append(qwen_response[index][prev_model_name][0])
        if response_list[0] == response_list[1]:
            item[model_name] = response_list[0]
            results.append(item)
            continue
        elif response_list[0] != item["Answer"] and response_list[1] != item["Answer"]:
            item[model_name] = response_list[0]
            results.append(item)
            continue
        
        ades = aitem_des['r1-avqa_audio_des_split5_1024_time']
        sdes = aitem_subtitle['r1-avqa_audio_subtitle_split5_1024']
        vtimes = []
        vdess = []
        for time_mname, item_name, vdes_mname, vdes_iname in video_des_json_name:
            time_path = os.path.join(time_dir,vname, f"{vname}_{time_mname}.json")
            with open(time_path, 'r', encoding='utf-8') as file:
                time_des_json = json.load(file)
            if item_name == "":
                vtimes.append(time_des_json[question])
            else:
                vtimes.append(time_des_json[question][item_name])
            
            time_path = os.path.join(time_dir,vname, f"{vname}_{vdes_mname}.json")
            with open(time_path, 'r', encoding='utf-8') as file:
                time_des_json = json.load(file)
            vdess.append(time_des_json[vdes_iname])


        question = item['Question']
        option_group = item["Choice"]
        abc_str = ""
        option_str = "\n"
        for i in range(len(option_group)-1):
            abc_str = abc_str + abc[i]+", "
            option_str = option_str + option_group[i] + "\n"
        abc_str = abc_str + "or "+abc[len(option_group)-1]
        option_str = option_str + option_group[len(option_group)-1] + "\n"

        text = ""
        all_merge_time = []
        for i in range(len(response_list)):
            time_des = vtimes[i]
            if isinstance(time_des, dict):
                time_des =[time_des]
            merge_time, merge_reason = merge(time_des,aitem_time) 
            text= text + f"Executor {str(i+1)}'s Answer: {response_list[i]}\n"
            all_merge_time = all_merge_time + merge_time

        unique = [list(t) for t in set(tuple(item) for item in all_merge_time)]
        unique.sort(key=lambda x: x[0])
        content = []
        extract_fps =1.0
        all_time = aitem_subtitle["subtitle_time"]
        for i in range(len(unique)):
            start_time = unique[i][0]
            end_time = unique[i][1]

            key_frames, audio_bytes_segments, time_segments = extract_frames_and_audio(video_path, sound_path, extract_fps=extract_fps, time_range=unique[i])
            time_str=""
            for j in range(len(key_frames)):
                time_str = time_str + str(time_segments[j]) + ", " 
                content.append(types.Part.from_bytes(data=key_frames[j], mime_type='image/jpg'))
            content.append(types.Part.from_bytes(data=audio_bytes_segments, mime_type='audio/mp3'))
            
            cur_vdes, cur_sds, cur_ads = find_subtitle(all_time,ades,sdes,vdess,start_time,end_time)


            all_subtitle=f"{float(start_time):.0f}s–{float(end_time):.0f}s: {aitem_subtitle['r1-avqa_audio_subtitle_split5_1024'][i][0]} "
            inp_select = f"\nThere are the key frames at time points {time_str}along with the corresponding audio.\n"\
                f"Subtitle: {cur_sds}\n"\
                f"Audio description: {cur_ads}\n" 
                
            content.append(inp_select)
            


        content.append(text)
        content.append("For this multiple-choice question, there are two different answers. Based on the all video frames, audio, subtitle, and audio descriptions, determine which option is correct. Respond with only the letter (" + abc_str + ") of the correct option."\
                       f"Question: {question}{option_str}\n")

        conversation = [
            {
                "role": "user",
                "content": content,
            },
        ]
        a=1
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model= "gemini-2.0-flash",
                    contents= content
                )
                break  # 如果成功，就跳出循环
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # 可选：等待一秒再重试
                else:
                    raise  # 最后一次还失败就抛出异常

        item[model_name] = response.text
        print("Response:", response.text)
        print("correct:", item["Answer"])

        results.append(item)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
print("over")