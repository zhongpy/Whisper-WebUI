import os
import argparse
import requests
import json

from modules.whisper_inference_console import ConsoleWhisperInference
from modules.faster_whisper_inference_console import ConsoleFasterWhisperInference
from modules.nllb_inference_console import ConsoleNLLBInference
from modules.whisper_data_class import *


class Auto:
    def __init__(self, args):
        self.args = args
        self.whisper_model="medium"
        self.device="cuda"
        self.NLLB_model="facebook/nllb-200-1.3B"
        self.videoLists=None
        self.whisper_inf=None
        self.nllb_inf=None
        self.process_info={}
        self.translate_languages={"en":"English","ja":"Japanese","zh_hant":"Chinese (Traditional)","ko":"Korean","vi":"Vietnamese","tha":"Thai","ind":"Indonesian","hi":"Hindi","ar":"Modern Standard Arabic","de":"German","fr":"French","it":"Italian","ru":"Russian","es":"Spanish","zsm":"Standard Malay"}
        

    def Load_whisper(self):
        if self.whisper_inf==None:
            self.whisper_inf = ConsoleWhisperInference() if self.args.disable_faster_whisper else ConsoleFasterWhisperInference()
            if isinstance(self.whisper_inf, ConsoleWhisperInference):
                print("Use Faster Whisper implementation")
            else:
                print("Use Open AI Whisper implementation")
            print(f"Device \"{self.whisper_inf.device}\" is detected")
        self.whisper_inf.update_model(self.whisper_model,self.device)

    def Load_NLLB(self):
        if self.nllb_inf==None:
            self.nllb_inf = ConsoleNLLBInference()
        self.nllb_inf.update_model(self.NLLB_model)

    def save_dict_to_txt(self,data, folder_path, file_name):
        """
        保存 Python 字典到指定的 txt 文件中。

        Args:
            data (dict): 要保存的字典数据。
            folder_path (str): 文件夹路径。
            file_name (str): 文件名。

        Returns:
            bool: 如果保存成功，返回 True；否则返回 False。
        """
        try:
            # 检查文件夹是否存在，如果不存在则创建
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # 拼接文件路径
            file_path = os.path.join(folder_path, file_name)

            # 将字典转换为 JSON 格式的字符串
            data_str = json.dumps(data)

            # 将 JSON 格式的字符串写入文本文件
            with open(file_path, "w") as f:
                f.write(data_str)

            print(f"字典已保存到 {file_path}")
            return True
        except Exception as e:
            print(f"保存失败：{e}")
            return False

    def load_dict_from_txt(self,file_path):
        """
        从指定的 txt 文件中读取字典数据。

        Args:
            file_path (str): 文件路径。

        Returns:
            dict: 读取到的字典数据，如果文件不存在或读取失败，返回空字典 {}。
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"文件 {file_path} 不存在")
                return {}

            # 读取文件内容
            with open(file_path, "r") as f:
                data_str = f.read()

            # 将 JSON 格式的字符串转换为 Python 字典
            my_dict = json.loads(data_str)
            return my_dict
        except Exception as e:
            print(f"读取失败：{e}")
            return {}


    def TranscribeVideo(self,folder_path,file_name,save_name):
        whisper_params=WhisperValues(
            model_size=self.whisper_model,
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
        self.whisper_inf.transcribe_file(folder_path,save_name,file_name,"SRT",False,*whisper_params)

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
            for episodeinfo in episodes:
                epid=episodeinfo['id']
                epurl=episodeinfo['videourl']
                zh_hansfile=self.whisper_model+"/zh_hans/"+str(epid)+".srt"
                if not self.whisper_model in self.process_info:
                    self.process_info[self.whisper_model]={}
                if not 'zh_hans' in self.process_info[self.whisper_model]:
                    self.process_info[self.whisper_model]['zh_hans']={}
                if not epid in self.process_info[self.whisper_model]['zh_hans'] or self.process_info[self.whisper_model]['zh_hans'][epid]==0:
                    self.process_info[self.whisper_model]['zh_hans'][epid]=0
                    result_str, result_file_path=self.TranscribeVideo(self.whisper_model,"zh_hans",epurl,str(epid))
                    print(result_str)
                    print(result_file_path)
                    self.process_info[self.whisper_model]['zh_hans'][epid]=1
                for lankey,lanvalue in self.translate_languages:
                    lankey_file=self.whisper_model+"/"+lankey_file+"/"+str(epid)+".srt"
                    if not lankey in self.process_info[self.whisper_model]:
                        self.process_info[self.whisper_model][lankey]={}
                    if not epid in self.process_info[self.whisper_model][lankey] or self.process_info[self.whisper_model][lankey][epid]==0:
                        self.process_info[self.whisper_model][lankey][epid]=0
                        gr_str, output_path=self.nllb_inf.translate_file(self.whisper_model,lankey,zh_hansfile,self.NLLB_model,"Chinese (Simplified)",lanvalue,False)
                        print(gr_str)
                        print(output_path)
                        self.process_info[self.whisper_model][lankey][epid]=1
            self.save_dict_to_txt(self.process_info,"autoprocess","processinfo.txt")
        else:
          print("API request failed!")

    def beginprocess(self):
        self.Load_whisper();
        self.Load_NLLB();
        self.process_info=self.load_dict_from_txt('./autoprocess/processinfo.txt')
        self.GetAllVideoList();



# Create the parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--disable_faster_whisper', type=bool, default=False, nargs='?', const=True, help='Disable the faster_whisper implementation. faster_whipser is implemented by https://github.com/guillaumekln/faster-whisper')
_args = parser.parse_args()

if __name__ == "__main__":
    app = Auto(args=_args)
    app.beginprocess()
