# XGC-AVis Code Documentation

## Overview

XGC-AVis is a multi-agent framework for enhancing audio-video temporal alignment for multimodal models. This repository provides a series of Python scripts for processing video and audio data, extracting relevant segments, and generating answers using different models (e.g., Gemini Flash, Qwen-Omni). The overall flow is organized into sequential steps to achieve the goal of localizing relevant video segments and providing model-based answers for the given queries.

## Directory Structure

1. **step1_transcribe_subtitle.py**: This script transcribes subtitles from audio files.
2. **step2_describe_audio_with_r1avqa.py**: Describes the audio content using the open-source `r1-avqa` model.
3. **step3_localize_relevant_segments_with_aria.py**: Identifies relevant time segments in the video based on provided questions, using the Aria planner.
4. **step3b_localize_relevant_segments_with_qwen_omni.py**: Identifies relevant time segments in the video using the Qwen Omni model.
5. **step4_answer_with_gemini_flash.py**: Uses the Gemini Flash model to generate answers based on the localized video segments and descriptions.
6. **step4b_answer_with_gemini_flash.py**: A variant of the above script for generating alternative answers.
7. **step5_Adjudication.py**: The final adjudication step where multiple outputs are processed and a final answer is selected based on the model outputs.


## How to Run

To run the entire process, execute the scripts in the following order:

1. **Step 1 - Transcribe subtitles**:

   ```bash
   python scripts/step1_transcribe_subtitle.py
   ```

2. **Step 2 - Describe the audio**:
   Use `r1-avqa` for audio description:

   ```bash
   python scripts/step2_describe_audio_with_r1avqa.py
   ```

3. **Step 3 - Localize relevant segments with Aria**:

   ```bash
   python scripts/step3_localize_relevant_segments_with_aria.py
   ```

4. **Step 3b - Localize relevant segments with Qwen Omni**:
   This step uses the Qwen Omni planner.

   ```bash
   python scripts/step3b_localize_relevant_segments_with_qwen_omni.py
   ```

5. **Step 4 - Answer with Gemini Flash**:

   ```bash
   python scripts/step4_answer_with_gemini_flash.py
   ```

6. **Step 4b - Answer with Gemini Flash**:

   ```bash
   python scripts/step4b_answer_with_gemini_flash.py
   ```

7. **Step 5 - Adjudication**:
   Final adjudication to determine the best answer.

   ```bash
   python scripts/step5_Adjudication.py
   ```

## Notes on Models Used

1. **Daily-Omni**:

   * We use **Daily-Omni** as a third-party library for this code. Since it is publicly available, it serves as the primary planner for video segment localization in this pipeline.
2. **Deepgram vs R1-AVQA**:

   * The original pipeline used **Deepgram** for audio transcription and analysis, but it requires a paid API. To avoid this, we have substituted it with the **R1-AVQA** model, an open-source alternative that serves the same function in the pipeline. The script `step2_describe_audio_with_r1avqa.py` demonstrates how to use this model.


