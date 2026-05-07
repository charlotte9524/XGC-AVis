import os
import json
import math
from google import genai
from google.genai import types
import copy
import io
from pydub import AudioSegment
import cv2
import moviepy.editor as mp
import time
max_retries = 5

client = genai.Client(api_key='xx')

def frame_to_timecode(frame_idx, fps):
    total_seconds = frame_idx / fps
    m = int(total_seconds // 60)
    s = int(total_seconds % 60)
    return f"{m}:{s:02d}"
def extract_frames_and_audio(video_path, audio_path, extract_fps=1, time_range=None):
    start_time, end_time= time_range[0],time_range[1]
    # 打开视频
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps

    # 设置默认结束时间为视频总时长
    if end_time is None or end_time > total_duration:
        end_time = total_duration
    start_frame_idx = int(start_time * fps)
    end_frame_idx = int(end_time * fps)
    if end_frame_idx > total_frames:
        end_frame_idx = total_frames - 1

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

def merge(time_des,ades,sdes,vdes,video_duration,all_time):
    merge_time = []
    merge_reason = []
    merge_time.append(closest_multiple_of_5([time_des[0]["start_time"],time_des[0]["end_time"]],video_duration))
    if 'reasoning' not in time_des[0]:
        cur_reason=""
    else:
        cur_reason = f"{float(time_des[0]['start_time']):.0f}s–{float(time_des[0]['end_time']):.0f}s: "\
            f"{time_des[0]['reasoning']}"
    merge_reason.append(cur_reason)
    if len(time_des) > 1:
        for i in range(1, len(time_des)):
            if float(time_des[i]["start_time"])>=video_duration:
                continue
            cur_time = closest_multiple_of_5([time_des[i]["start_time"],time_des[i]["end_time"]],video_duration)
            if cur_time[0] < merge_time[-1][1]:
                merge_time[-1][1] = cur_time[1]
                if 'reasoning' not in time_des[i]:
                    cur_reason=""
                else:
                    cur_reason = f"{float(time_des[i]['start_time']):.0f}s–{float(time_des[i]['end_time']):.0f}s: "\
                                    f"{time_des[i]['reasoning']}"
                merge_reason[-1] =  merge_reason[-1] + cur_reason
            else:
                merge_time.append(cur_time)
                if 'reasoning' not in time_des[i]:
                    cur_reason=""
                else:
                    cur_reason = f"{float(time_des[i]['start_time']):.0f}s–{float(time_des[i]['end_time']):.0f}s: "\
                                    f"{time_des[i]['reasoning']}"
                merge_reason.append(cur_reason)
    merge_ades, merge_sdes, merge_vdes = [],[],[]
    segment_time_total = 0
    for i in range(len(merge_time)):
        segment_time_total += merge_time[i][1]-merge_time[i][0]
        cur_vdes = ""
        cur_ads = ""
        cur_sdes = ""
        for j in range(len(all_time)):
            if all_time[j][0] >= merge_time[i][0] and all_time[j][0] < merge_time[i][1]:
                cur_vdes += f"{all_time[j][0]}s-{all_time[j][1]}s: {vdes[j].replace('<|im_end|>','')} "
                cur_sdes += f"{all_time[j][0]}s-{all_time[j][1]}s: {sdes[j][0]} "
                cur_ads += f"{all_time[j][0]}s-{all_time[j][1]}s: {ades[j]} "
            elif all_time[j][1] > merge_time[i][0] and all_time[j][1] <= merge_time[i][1]:
                cur_vdes += f"{all_time[j][0]}s-{all_time[j][1]}s: {vdes[j].replace('<|im_end|>','')} "
                cur_sdes += f"{all_time[j][0]}s-{all_time[j][1]}s: {sdes[j][0]} "
                cur_ads += f"{all_time[j][0]}s-{all_time[j][1]}s: {ades[j]} "
            else:
                continue
        
        merge_sdes.append(cur_sdes)
        merge_ades.append(cur_ads)
        merge_vdes.append(cur_vdes)
    return merge_time, merge_reason, merge_vdes, merge_ades, merge_sdes, segment_time_total
if __name__ == "__main__":
    json_name = 'qa.json'
    cut_video_json_path = './Daliy-Omni/' + json_name
    with open(cut_video_json_path, 'r', encoding='utf-8') as file:
        cut_video_json = json.load(file)

    with open('./Daliy-Omni/result/qa.json_r1-avqa_audio_des_split5_1024_time.json', 'r', encoding='utf-8') as file:
        ades_split5_json = json.load(file)
    with open("./Daliy-Omni/result/qa.json_r1-avqa_audio_subtitle_split5_1024.json", 'r', encoding='utf-8') as file:
        subtitle_split5_json = json.load(file)

    time_mname = 'Aria'
    time_step = "video_time_range"
    time_dir = f'./Daliy-Omni/Agent_result/memory'

    video_dir = "./Daliy-Omni/Videos"
    model_name = "gemini-2.0-flash"
    abc = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    video_dir = './Daliy-Omni/Videos'


    result_path = f'./Daliy-Omni/Agent_result/Qwen2.5-Omni/{time_mname}_{time_step}_{model_name}+subtitle.json'
    results = []
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as file:
            results = json.load(file)
    for index in range(len(results), len(cut_video_json)):#
        item = cut_video_json[index]
        aitem_des = ades_split5_json[index]
        aitem_subtitle = subtitle_split5_json[index]
        video_duration = int(item["video_duration"][:-1])

        vname = item['video_id']
        video_duration = int(item["video_duration"][:-1])
        video_path = os.path.join(video_dir, vname, vname+"_video.mp4")
        sound_path = os.path.join(video_dir, vname, vname+"_audio.wav")

        time_path = os.path.join(time_dir,vname, f"{vname}_{time_mname}_{time_step}.json")
        with open(time_path, 'r', encoding='utf-8') as file:
            time_des_json = json.load(file)
        time_des = time_des_json[item['Question']]["time_segments"]
        if 'start_time' in time_des:
            time_des =[time_des]
        ades = aitem_des['r1-avqa_audio_des_split5_1024_time']
        sdes = aitem_subtitle['r1-avqa_audio_subtitle_split5_1024']
        vdes_path = os.path.join(time_dir,vname, f"{vname}_{time_mname}_video_des_split5.json")
        with open(vdes_path, 'r', encoding='utf-8') as file:
            vdes_json = json.load(file)
        vdes = vdes_json["video_des_split5"]
        time_des = sorted(time_des, key=lambda x: float(x["start_time"]))
        merge_time, merge_reason, merge_vdes, merge_ades, merge_sdes, segment_time_total = merge(time_des,ades, sdes,vdes,video_duration,aitem_des["all_time"])

        if segment_time_total>= 30:
            extract_fps = 2
        else:
            extract_fps = 1
        question = item['Question']
        option_group = item["Choice"]
        abc_str = ""
        option_str = "\n"
        for i in range(len(option_group)-1):
            abc_str = abc_str + abc[i]+", "
            option_str = option_str + option_group[i] + "\n"
        abc_str = abc_str + "or "+abc[len(option_group)-1]
        option_str = option_str + option_group[len(option_group)-1] + "\n"


        inp_select = "Please carefully read the questions related to audio-visual perception, understanding, and reasoning abilities, and carefully watch the above video clip, the audio clip, the audio description and the video description. " +\
        "Based on your observations, select the best option that accurately addresses the question." +\
        "Respond with only the letter ("+abc_str+") of the correct option, followed by 'Reason:' and your reasoning.\n" +\
        "Question: " + question+option_str+"Answer: "

        content = []
        for ind in range(len(merge_time)):
            time_range = merge_time[ind]
            reason = merge_reason[ind]
            cur_vdes = merge_vdes[ind].replace("\n","")
            cur_ads = merge_ades[ind].replace("\n","")
            cur_sds = merge_sdes[ind].replace("\n","")

            key_frames, audio_bytes_segments, time_segments = extract_frames_and_audio(video_path, sound_path, extract_fps=extract_fps, time_range=time_range)
            time_str=""
            for i in range(len(key_frames)):
                time_str = time_str + str(time_segments[i]) + ", " 
                content.append(types.Part.from_bytes(data=key_frames[i], mime_type='image/jpg'))
            content.append(types.Part.from_bytes(data=audio_bytes_segments, mime_type='audio/mp3'))
            
            text = f"The above frames and audio are extracted from {time_str}.\n"\
                f"Subtitle: {cur_sds}\n"\
                f"Audio description: {cur_ads}\n" \
                f"Video description: {reason}\n"       
            content.append(text) #
        content.append(inp_select)
                
        a=1
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model= model_name,
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

        results.append(item)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
    print("over")