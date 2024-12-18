import os
import argparse
import yaml
import requests
import json

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, WHISPER_MODELS_DIR,
                                 INSANELY_FAST_WHISPER_MODELS_DIR, NLLB_MODELS_DIR, DEFAULT_PARAMETERS_CONFIG_PATH,
                                 UVR_MODELS_DIR, I18N_YAML_PATH)
from modules.utils.files_manager import load_yaml
from modules.whisper.whisper_factory import WhisperFactory
from modules.translation.nllb_inference import NLLBInference
from modules.ui.htmls import *
from modules.utils.cli_manager import str2bool
from modules.utils.youtube_manager import get_ytmetas
from modules.translation.deepl_api import DeepLAPI
from modules.whisper.data_classes import *


class Auto:
    def __init__(self, args):
        self.args = args
        self.whisper_inf = WhisperFactory.create_whisper_inference(
            whisper_type=WhisperImpl.WHISPER.value,
            whisper_model_dir=WHISPER_MODELS_DIR,
            faster_whisper_model_dir=FASTER_WHISPER_MODELS_DIR,
            insanely_fast_whisper_model_dir=INSANELY_FAST_WHISPER_MODELS_DIR,
            uvr_model_dir=UVR_MODELS_DIR,
            output_dir=OUTPUT_DIR,
        )
        self.nllb_inf = NLLBInference(
            model_dir=NLLB_MODELS_DIR,
            output_dir=os.path.join(OUTPUT_DIR, "translations")
        )
        self.deepl_api = DeepLAPI(
            output_dir=os.path.join(OUTPUT_DIR, "translations")
        )
        self.default_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        print(f"Use \"{self.args.whisper_type}\" implementation\n"
              f"Device \"{self.whisper_inf.device}\" is detected")

        self.whisper_model="large-v3"
        self.NLLB_model="facebook/nllb-200-3.3B"
        self.device="cuda"
        self.videoLists=None
        self.process_info={}
        self.translate_languages={"en":"English","ja":"Japanese","zh_hant":"Chinese (Traditional)","ko":"Korean","vi":"Vietnamese","tha":"Thai","ind":"Indonesian","hi":"Hindi","ar":"Modern Standard Arabic","de":"German","fr":"French","it":"Italian","ru":"Russian","es":"Spanish","zsm":"Standard Malay"}
        #{"en":"English","ja":"Japanese","zh_hant":"Chinese (Traditional)"}
        self.LoadModel();

    def LoadModel(self):
        self.whisper_inf.update_model(self.whisper_model,self.whisper_inf.current_compute_type,self.progress);

    def create_pipeline_inputs_console(self):
        whisper_params = self.default_params["whisper"]
        vad_params = self.default_params["vad"]
        diarization_params = self.default_params["diarization"]
        uvr_params = self.default_params["bgm_separation"]

        # 扁平化 Whisper 参数
        whisper_list = [
            whisper_params.get("model_size", "large-v3"),
            whisper_params.get("lang", "chinese"),
            whisper_params.get("is_translate", False),
            whisper_params.get("beam_size", 5),
            whisper_params.get("log_prob_threshold", -1.0),
            whisper_params.get("no_speech_threshold", 0.6),
            whisper_params.get("compute_type", "float16"),
            whisper_params.get("best_of", 5),
            whisper_params.get("patience", 1.0),
            whisper_params.get("condition_on_previous_text", True),
            whisper_params.get("prompt_reset_on_temperature", 0.5),
            whisper_params.get("initial_prompt", None),
            whisper_params.get("temperature", 0.0),
            whisper_params.get("compression_ratio_threshold", 2.4),
            whisper_params.get("length_penalty", 1.0),
            whisper_params.get("repetition_penalty", 1.0),
            whisper_params.get("no_repeat_ngram_size", 0),
            whisper_params.get("prefix", None),
            whisper_params.get("suppress_blank", True),
            whisper_params.get("suppress_tokens", [-1]),
            whisper_params.get("max_initial_timestamp", 1.0),
            whisper_params.get("word_timestamps", False),
            whisper_params.get("prepend_punctuations", "\"'“¿([{-"),
            whisper_params.get("append_punctuations", "\"'.。,，!！?？:：”)]}、"),
            whisper_params.get("max_new_tokens", None),
            whisper_params.get("chunk_length", 30),
            whisper_params.get("hallucination_silence_threshold", None),
            whisper_params.get("hotwords", None),
            whisper_params.get("language_detection_threshold", None),
            whisper_params.get("language_detection_segments", 1),
            whisper_params.get("batch_size", 24)
        ]

        # 扁平化 VAD 参数
        vad_list = [
            vad_params.get("vad_filter", False),
            vad_params.get("threshold", 0.5),
            vad_params.get("min_speech_duration_ms", 250),
            vad_params.get("max_speech_duration_s", float("inf")),
            vad_params.get("min_silence_duration_ms", 2000),
            vad_params.get("speech_pad_ms", 400)
        ]

        # 扁平化 Diarization 参数
        diarization_list = [
            diarization_params.get("is_diarize", False),
            diarization_params.get("device", "cuda"),
            diarization_params.get("hf_token", "")
        ]

        # 扁平化 BGM Separation 参数
        bgm_sep_list = [
            uvr_params.get("is_separate_bgm", False),
            uvr_params.get("model_size", "UVR-MDX-NET-Inst_HQ_4"),
            uvr_params.get("device", "cuda"),
            uvr_params.get("segment_size", 256),
            uvr_params.get("save_file", False),
            uvr_params.get("enable_offload", True)
        ]

        dd_model = whisper_params["model_size"]
        dd_lang = whisper_params["lang"]
        cb_translate = whisper_params["is_translate"]

        pipeline_inputs = [dd_model, dd_lang, cb_translate] + whisper_list + vad_list + diarization_list + bgm_sep_list

        file_format = whisper_params.get("file_format", "SRT")
        add_timestamp = whisper_params.get("add_timestamp", False)

        return pipeline_inputs, file_format, add_timestamp



    def progress(self,total,desc="",position=None):
        return

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


    def TranscribeVideo(self,rootfolder,lanfolder,file_name,save_name):
        pipeline_inputs, file_format, add_timestamp=self.create_pipeline_inputs_console()
        print("[DEBUG] Pipeline List:")
        print(pipeline_inputs)
        print("[DEBUG] File Format:", file_format)
        print("[DEBUG] Add Timestamp:", add_timestamp)

        # 调用 from_list 时打印输入
        params = TranscriptionPipelineParams.from_list(pipeline_inputs)
        print("[DEBUG] TranscriptionPipelineParams parsed successfully.")
        return self.whisper_inf.transcribe_file_web(rootfolder,lanfolder,save_name,file_name,file_format,add_timestamp,self.progress,pipeline_inputs)

    def GetAllVideoList(self):
        url = "https://dyhaojiu.jaxczs.cn/api/video/getAllVideoList"
        response = requests.get(url)
        if response.status_code == 200:
            # Get the data in JSON format
            self.videoLists = response.json()
            #print(self.videoLists)
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
                if epurl=="":
                    continue
                epidkey=str(epid)
                zh_hansfile="subtitle/"+self.whisper_model+"/zh_hans/"+str(epid)+".srt"
                if not self.whisper_model in self.process_info:
                    self.process_info[self.whisper_model]={}
                if not 'zh_hans' in self.process_info[self.whisper_model]:
                    self.process_info[self.whisper_model]['zh_hans']={}
                if not epidkey in self.process_info[self.whisper_model]['zh_hans'] or self.process_info[self.whisper_model]['zh_hans'][epidkey]==0:
                    self.process_info[self.whisper_model]['zh_hans'][epidkey]=0
                    if self.TranscribeVideo("subtitle/"+self.whisper_model,"zh_hans",epurl,str(epid)):
                        self.process_info[self.whisper_model]['zh_hans'][epidkey]=1
                    else:
                        print("Transcribe Failed:Videoid:"+str(vid)+" epid:"+str(epid))
                for lankey,lanvalue in self.translate_languages.items():
                    lankey_file="subtitle/"+self.whisper_model+"/"+lankey+"/"+str(epid)+".srt"
                    if not lankey in self.process_info[self.whisper_model]:
                        self.process_info[self.whisper_model][lankey]={}
                    if not epidkey in self.process_info[self.whisper_model][lankey] or self.process_info[self.whisper_model][lankey][epidkey]==0:
                        self.process_info[self.whisper_model][lankey][epidkey]=0
                        if self.nllb_inf.translate_file_web("subtitle/"+self.whisper_model,lankey,zh_hansfile,self.NLLB_model,"Chinese (Simplified)",lanvalue,1000,False,self.progress):
                            self.process_info[self.whisper_model][lankey][epidkey]=1
                        else:
                            print("Translate Failed:Videoid:"+str(vid)+" epid:"+str(epid)+" lan:"+lanvalue)
                self.save_dict_to_txt(self.process_info,"autoprocess","processinfo.txt")
        else:
          print("API request failed!")

    def beginprocess(self):
        self.process_info=self.load_dict_from_txt('./autoprocess/processinfo.txt')
        self.GetAllVideoList();



# Create the parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--whisper_type', type=str, default=WhisperImpl.WHISPER.value,
                    choices=[item.value for item in WhisperImpl],
                    help='A type of the whisper implementation (Github repo name)')
_args = parser.parse_args()

if __name__ == "__main__":
    app = Auto(args=_args)
    app.beginprocess()
