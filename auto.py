import os
import argparse
import requests

from modules.whisper_inference_console import ConsoleWhisperInference
from modules.faster_whisper_inference_console import ConsoleFasterWhisperInference
from modules.nllb_inference_console import ConsoleNLLBInference
from modules.whisper_data_class import *


class Auto:
    def __init__(self, args):
        self.args = args
        self.videoLists=None
        #self.whisper_inf = ConsoleWhisperInference() if self.args.disable_faster_whisper else ConsoleFasterWhisperInference()
        #if isinstance(self.whisper_inf, ConsoleWhisperInference):
        #    print("Use Faster Whisper implementation")
        #else:
        #    print("Use Open AI Whisper implementation")
        #print(f"Device \"{self.whisper_inf.device}\" is detected")
        #self.nllb_inf = ConsoleNLLBInference()

    def Load_whisper(self):
        self.whisper_inf.update_model(self.args.whisper_model,"cuda")

    def Load_NLLB(self):
        self.nllb_inf.update_model(self.args.NLLB_model)

    def TranscribeVideo(self,file_name):
        whisper_params=WhisperValues(
            model_size=self.args.whisper_model,
            lang="Automatic Detection",
            is_translate=False,
            beam_size=1,
            log_prob_threshold=-1,
            no_speech_threshold=0.6,
            compute_type="float16",
            best_of=5,
            patience=1,
            condition_on_previous_text=true,
            initial_prompt=None
            )
        self.whisper_inf.transcribe_file(file_name,"SRT",False,*whisper_params)

    def GetAllVideoList(self):
        url = "https://dyhaojiu.jaxczs.cn/api/video/getAllVideoList"
        response = requests.get(url)
        if response.status_code == 200:
            # Get the data in JSON format
            self.videoLists = response.json()
            print(self.videoLists)
            for videoinfo in self.videoLists:
                self.GetVideoEpisodes(videoinfo['id'])
        else:
          print("API request failed!")

    def GetVideoEpisodes(self,vid):
        url = "https://dyhaojiu.jaxczs.cn/api/video/getVideoEpisode?vid="+str(vid)
        response = requests.get(url)
        if response.status_code == 200:
            # Get the data in JSON format
            episodes = response.json()
            print(episodes)
        else:
          print("API request failed!")

    def beginprocess(self):
        #self.Load_whisper();
        #self.Load_NLLB();
        self.GetAllVideoList();



# Create the parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--disable_faster_whisper', type=bool, default=False, nargs='?', const=True, help='Disable the faster_whisper implementation. faster_whipser is implemented by https://github.com/guillaumekln/faster-whisper')
parser.add_argument('--whisper_model', type=str, default='medium', help='Whisper Model')
parser.add_argument('--NLLB_model', type=str, default='facebook/nllb-200-1.3B', help='NLLB Model')
parser.add_argument('--folder', type=str, default=None, help='folder')
_args = parser.parse_args()

if __name__ == "__main__":
    app = Auto(args=_args)
    app.beginprocess()
