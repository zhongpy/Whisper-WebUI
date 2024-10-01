from modules.whisper.whisper_factory import WhisperFactory
from modules.whisper.whisper_parameter import WhisperValues
from modules.utils.paths import WEBUI_DIR
from test_config import *

import requests
import pytest
import gradio as gr
import os


@pytest.mark.parametrize("whisper_type", ["whisper", "faster-whisper", "insanely_fast_whisper"])
def test_transcribe(whisper_type: str):
    audio_path_dir = os.path.join(WEBUI_DIR, "tests")
    audio_path = os.path.join(audio_path_dir, "jfk.wav")
    if not os.path.exists(audio_path):
        download_file(TEST_FILE_DOWNLOAD_URL, audio_path_dir)

    whisper_inferencer = WhisperFactory.create_whisper_inference(
        whisper_type=whisper_type,
    )
    print("Device : ", whisper_inferencer.device)

    hparams = WhisperValues(
        model_size=TEST_WHISPER_MODEL,
    ).as_list()

    subtitle_str, file_path = whisper_inferencer.transcribe_file(
        [audio_path],
        None,
        "SRT",
        False,
        gr.Progress(),
        *hparams,
    )

    assert isinstance(subtitle_str, str) and subtitle_str
    assert isinstance(file_path[0], str) and file_path

    whisper_inferencer.transcribe_youtube(
        TEST_YOUTUBE_URL,
        "SRT",
        False,
        gr.Progress(),
        *hparams,
    )
    assert isinstance(subtitle_str, str) and subtitle_str
    assert isinstance(file_path[0], str) and file_path

    whisper_inferencer.transcribe_mic(
        audio_path,
        "SRT",
        False,
        gr.Progress(),
        *hparams,
    )
    assert isinstance(subtitle_str, str) and subtitle_str
    assert isinstance(file_path[0], str) and file_path


def download_file(url, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = url.split("/")[-1]
    file_path = os.path.join(save_dir, file_name)

    response = requests.get(url)

    with open(file_path, "wb") as file:
        file.write(response.content)

    print(f"File downloaded to: {file_path}")
