# FJ Chat to Speech

<img alt="FJ Chat to Speech" src="./docs/app_0.png" width="700">

FJ Chat to Speech is an open-source desktop application that converts live chat messages from YouTube and Twitch streams into real-time speech.

## <a href="https://github.com/facejungle/fj_chat_to_speech/releases/latest/" target="_blank">Download</a>

The application combines:

- YouTube and Twitch chat integration
- `Silero` text-to-speech models (local inference)
- `Detoxify` toxicity filtering (local inference)
- Custom spam filtering tools

## Key Features

- Voice selection (English and Russian)
- Adjustable speech volume
- Adjustable speech speed
- Pronunciation of numbers
- Optional translation of messages before speech
- Stop-word list editor
- Toxicity filtering (Detoxify)
- English and Russian interface
- Free and open-source

### YouTube Streams

To connect to a broadcast, simply paste the link to the active live broadcast and click "Connect"

### Twitch Streams

For Twitch, simply enter the channel name (or URL) of an active stream. FJ Chat to Speech will connect to Twitch’s chat service using your Client ID and start reading incoming chat messages. (Beforehand, you must register a Twitch app and copy the Client ID from the Twitch Developer Console into the program.) The interface will handle reconnects if the stream goes offline and comes back.

- <a href="https://github.com/facejungle/fj_chat_to_speech/wiki/Twitch-CLIENT-ID" target="_blank">How to create Client ID</a>

## Launch from source code

```bash
git clone https://github.com/facejungle/fj_chat_to_speech.git
cd fj_chat_to_speech

pip install -r torch.requirements.txt
pip install -r requirements.txt

python main.py
```

## Build

```bash
python build.py
cd dist
ls
```

## Thanks for <a href="https://github.com/snakers4/silero-models/" target="_blank">Silero</a> and <a href="https://github.com/unitaryai/detoxify" target="_blank">Detoxify</a>

- **Silero Models**: We use Silero’s open-source TTS models, known for natural-sounding voices and fast CPU performance
- **Detoxify**: Toxicity filtering is powered by the Detoxify library, which provides pretrained models to flag and filter toxic comments
