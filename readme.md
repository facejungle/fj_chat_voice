# FJ Chat to Speech

<img alt="FJ Chat to Speech" src="./docs/app_0.png" width="700">

This is a program for real-time chat voiceover using the Silero model.

It uses the Google API to retrieve messages from YouTube chat and converts them to speech. The key acquisition method is described in the instructions.
The first launch may take 1-2 minutes, as the program downloads the model and caches it for subsequent launches.

- The program is completely free.
- The program interface is Russian and English, and it also supports voiceover in Russian and English.

Implemented features:

- Voice selection
- Spam filter and stop words list
- Adjustable voiceover volume
- Adjustable voiceover speed
- Adjustable number of messages in the queue
- Speak numbers in messages
- Translation of chat messages into speech language

## Text-to-speech for YouTube streams

- Generating [Google API Keys](https://github.com/facejungle/fj_chat_voice/wiki/Google-API-Keys)
- The Google API has a limit on the number of requests per day, typically quota equal 10,000. View your quota: [Google Console: Quotas](https://console.cloud.google.com/iam-admin/quotas)

## Text-to-speech for Twitch streams

- Generating [Twitch CLIENT ID](https://github.com/facejungle/fj_chat_voice/wiki/Twitch-CLIENT-ID)

## Launch from source code

```
git clone https://github.com/facejungle/fj_chat_voice.git
cd fj_chat_voice

pip install -r torch.requirements.txt
pip install -r requirements.txt

python main.py
```

## Build

```
python build.py
```

## Thanks for <a href="https://github.com/snakers4/silero-models/" target="_blank">Silero</a> :)
