import os
import json
import torch
import torchaudio
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# Load model
model_name = "mispeech/r1-aqa"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

wait_list =[
    ['qa.json', './Daily-Omni/Videos']
]
model_name = "r1-avqa_audio_subtitle_split5_1024"
abc = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
json_path = './Daily-Omni'

for wait_json in wait_list:
    json_name, video_dir = wait_json
    data_path = os.path.join(json_path, json_name)
    result_path = f'./Daily-Omni/result/subtitle/{json_name}_{model_name}.json'
    results = []
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as file:
            results = json.load(file)

    with open(f'./Daily-Omni/result/{json_name}_{model_name}.json', 'r', encoding='utf-8') as file:
        old_result = json.load(file)
    with open(os.path.join(json_path, json_name), 'r', encoding='utf-8') as file:
        cut_video_json = json.load(file)

    for index in range(len(results), len(cut_video_json)):
        item = cut_video_json[index]
        vname = item['video_id']
        wav_path = os.path.join(video_dir, vname, vname+'_audio.wav')

        waveform, sr = torchaudio.load(wav_path)
        sampling_rate=16000
        if sr != sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)(waveform)
        else:
            results.append(old_result[index])
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            continue

        video_duration = int(len(waveform[0])/16000)
        # question = item['Question']
        st = 0
        all_time = []
        if video_duration < 5:
            all_time = [[0,video_duration]]
        else:
            while st + 5 <= video_duration:
                all_time.append([st, st+5])
                st = st + 5
        all_time[-1][1] = video_duration
        item["all_time"] = all_time
        item[model_name] = []
        for itime in range(len(all_time)):
            time_range = all_time[itime]
            start_sample = int(time_range[0] * sampling_rate)  # 6秒对应的样本索引
            if itime == len(all_time)-1:
                end_sample = len(waveform[0])
            else:
                end_sample = int(time_range[1] * sampling_rate)  # 10秒对应的样本索引
            audio_segment = waveform[0][start_sample:end_sample]

            audios = [audio_segment.numpy()]
            audio_duration = int(time_range[1]-time_range[0])
            # inp = f"Please describe in detail what happens in the audio, including sounds, music, speech, details of the sounds, and emotions conveyed. "\
            #     "Please ensure that your response does not include any references to time, such as seconds, timestamps, or durations."

            inp = "Please transcribe the speech in the audio clip. "\
                "Extract all the words spoken by the individuals and translate any non-English parts into English. "\
                "Please indicate the speaker's gender. "\
                "If there is no speech in the audio, respond with No speech detected."\
                "Do not include any references to time, such as seconds, timestamps, or durations, in your response. "\
                "Do not respond using list or dictionary formats. Instead, write your response in full sentences, such as: "\
                "“A female voice says: '...'. Then a male voice responds: '...'."


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


            item[model_name].append(response[0])
        results.append(item)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        