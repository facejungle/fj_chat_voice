# FJ Chat Voice - Text to Speech

![alt text](./docs/app_0.png)

## Google API Key

### 1. Create new project

Go to Google Console: https://console.cloud.google.com/
![alt text](./docs/console_project_0.png)
![alt text](./docs/console_project_1.png)

### 2. Enable YouTube Data API v3

https://console.cloud.google.com/apis/api/youtube.googleapis.com/

### 3. Create API Key

Google Console > APIs & Services > Credentials > Create credentials > API key
![alt text](./docs/console_api_key_0.png)
![alt text](./docs/console_api_key_1.png)
![alt text](./docs/console_api_key_2.png)

# Build

## Linux

```
sudo apt update && sudo apt install python3-tk
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install pyinstaller pyinstaller-versionfile
python build.py
```

## Windows

Uncomment tkinter in requirements.txt

```
pip install pyinstaller pyinstaller-versionfile
python build.py
```

# Thanks for Silero https://github.com/snakers4/silero-models/ :)
