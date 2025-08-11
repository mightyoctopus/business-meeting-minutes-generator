# business-meeting-minutes-generator
Business meeting generator that transcribes a mp3 audio format into text and generate minutes of meeting from the transcribed text. Whisper-1 used for transcribing, meta-llama/Llama-3.1-8B-Instruct used for minutes of meeting gen (quantized in the code). 

## How to Use:
1. Upload an mp3 file of business meeting record to the app, using the upload button
2. Upon completion of the file upload, click on Transcribe.
3. It will process the audio file transcription using the whisper-1 model and then passed to a local ai model LLAMA to
   generate the minutes of meeting from the transcribed result.
4. The result will show up below the button. The generation might take a minute or 2 as it needs to process in multi level layers.

## Prerequisites:
- Audio Model: OpenAI Whisper-1
- Text Model: meta-llama/Llama-3.1-8B-Instruct (from Hugging Face)
- Others: Please refer to the requirements.txt file.
