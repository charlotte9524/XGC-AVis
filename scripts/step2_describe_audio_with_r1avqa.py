import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import torch
import torchaudio
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# import debugpy

# # 启动debugpy并监听指定地址和端口
# debugpy.listen(('127.0.0.1', 61345))

# # 输出启动成功信息，告诉用户可以开始连接
# print("Debugpy server is listening on 127.0.0.1:61345. Please connect your debugger.")

# # 等待VS Code或其他调试客户端的连接
# debugpy.wait_for_client()

# # 在调试客户端连接后继续执行后面的代码
# print("Debugger is connected, starting inference...")

# Load model
model_name = "mispeech/r1-aqa"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

wait_list =[
    ['qa.json', './Daily-Omni']
]
model_name = "r1-avqa_audio_des_split5_1024_time"
abc = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
audio_dir = './Daily-Omni/Videos'
json_path = './Daily-Omni'

for wait_json in wait_list:
    json_name, video_dir = wait_json
    data_path = os.path.join(json_path, json_name)
    result_path = f'./Daily-Omni/result/{json_name}_{model_name}.json'
    results = []
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as file:
            results = json.load(file)

    with open(os.path.join(json_path, json_name), 'r', encoding='utf-8') as file:
        cut_video_json = json.load(file)
    for item in cut_video_json[len(results):]:
        vname = item['video_id']
        wav_path = os.path.join(audio_dir, vname, vname+'_audio.wav')

        waveform, sampling_rate = torchaudio.load(wav_path)
        if sampling_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(waveform)

        video_duration = int(len(waveform[0])/16000)
        # question = item['Question']
        st = 0
        all_time = []
        while st + 5 <= video_duration:
            all_time.append([st, st+5])
            st = st + 5
        all_time[-1][1] = video_duration
        item["all_time"] = all_time
        item[model_name] = []
        for itime in range(len(all_time)):
            time_range = all_time[itime]
            start_sample = int(time_range[0] * sampling_rate)  # 6秒对应的样本索引
            end_sample = int(time_range[1] * sampling_rate)  # 10秒对应的样本索引
            audio_segment = waveform[0][start_sample:end_sample]

            audios = [audio_segment.numpy()]
            audio_duration = int(time_range[1]-time_range[0])
            inp = f"Please describe in detail what happens in the audio, including sounds, music, speech, details of the sounds, and emotions conveyed. "\
                "Please ensure that your response does not include any references to time, such as seconds, timestamps, or durations."

            message = [
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": wav_path},
                    {"type": "text", "text": inp}
                ]}
            ]
            texts = processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

            # Process
            inputs = processor(text=texts, audios=audios, sampling_rate=16000, return_tensors="pt", padding=True).to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids = generated_ids[:, inputs.input_ids.size(1):]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            print(response[0])

            item[model_name].append(response[0])
        results.append(item)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        