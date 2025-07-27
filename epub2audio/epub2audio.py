import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import ebooklib
import torchaudio
from bs4 import BeautifulSoup
from ebooklib import epub
from loguru import logger
from pydub import AudioSegment, silence

sys.path.append("third_party/Matcha-TTS")
try:
    from vllm import ModelRegistry

    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.common import set_all_random_seed
    from cosyvoice.utils.file_utils import load_wav
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
except ImportError as e:
    logger.error(
        f"Failed to import CosyVoice2 modules. Please ensure 'third_party/Matcha-TTS' is correctly set up and dependencies are installed. Error: {e}"
    )
    # Exit or handle gracefully if core components are missing
    sys.exit(1)

MODEL_PATH = "pretrained_models/CosyVoice2-0.5B"

BASE_DIR = Path("/mnt/c/Users/guozr/Documents/CosyVoice")
EPUB_PATH = BASE_DIR / "地下室手记.epub"

TXT_DIR = BASE_DIR / "epub_res"
AUDIO_DIR = BASE_DIR / "generated_audio"
ZERO_SHOT_PROMPT_DIR = BASE_DIR / "zero_shot_prompts"


MP4_PATH = BASE_DIR / "ladynana.mp4"
RAW_WAV_PATH = BASE_DIR / "laydnana.wav"

PROMPT_WAV_PATH = ZERO_SHOT_PROMPT_DIR / "laydnana_segment_000.wav"
PROMPT_TEXT = "中国的兄弟姐妹们你们好，我是俄罗斯娜娜。昨天晚上我的评论区炸锅了。"

# PROMPT_WAV_PATH = Path("/home/guozr/CODE/CosyVoice/asset/zero_shot_prompt.wav")
# PROMPT_TEXT = "希望你以后能够做的比我还好呦。"


FFMPEG_AUDIO_SAMPLE_RATE = 16000  # 音频采样率 (Audio sample rate)
FFMPEG_AUDIO_CHANNELS = 1  # 音频通道数 (Number of audio channels)
FFMPEG_AUDIO_CODEC = "pcm_s16le"  # 音频编码器 (Audio codec)

MIN_SILENCE_LEN = 500  # 最小静音长度 (ms) (Minimum silence length in ms)
SILENCE_THRESH = -40  # 静音阈值 (dBFS) (Silence threshold in dBFS)
KEEP_SILENCE = 200  # 保留静音长度 (ms) (Silence to keep around segments in ms)

NOISE_REDUCE_PROP_DECREASE = 0.9  # 噪声减少比例 (Noise reduction proportion)
NOISE_ESTIMATION_DURATION = (
    0.2  # 噪声估算时长 (秒) (Duration for noise estimation in seconds)
)


TXT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
ZERO_SHOT_PROMPT_DIR.mkdir(parents=True, exist_ok=True)


class AudiobookConverter:
    def __init__(self):
        self.cosyvoice = None  # CosyVoice2 model will be initialized lazily
        self.prompt_speech_16k = None  # Prompt speech for CosyVoice2

    @staticmethod
    def _run_ffmpeg_command(command_list, description=""):
        """运行ffmpeg命令并打印详细错误信息"""
        logger.info(" ".join(command_list))
        try:
            result = subprocess.run(
                command_list, check=True, capture_output=True, text=True
            )
            if description:
                logger.success(f"成功: {description}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg命令失败: {description}")
            logger.error(f"返回码: {e.returncode}")
            if e.stdout.strip():
                logger.error(f"标准输出: {e.stdout}")
            if e.stderr.strip():
                logger.error(f"标准错误: {e.stderr}")
            raise

    @staticmethod
    def mp42wav():
        if not MP4_PATH.exists():
            logger.error(f"MP4 文件未找到: {MP4_PATH}")
            return

        ffmpeg_convert_command = [
            "ffmpeg",
            "-i",
            str(MP4_PATH),
            "-vn",  # 无视频 (No video)
            "-acodec",
            FFMPEG_AUDIO_CODEC,  # PCM 16-bit little-endian
            "-ar",
            str(FFMPEG_AUDIO_SAMPLE_RATE),  # 16 kHz 采样率 (16 kHz sample rate)
            "-ac",
            str(FFMPEG_AUDIO_CHANNELS),  # 单声道音频 (Mono audio)
            "-y",  # 如果输出文件存在则覆盖 (Overwrite output file if it exists)
            str(RAW_WAV_PATH),
        ]

        AudiobookConverter._run_ffmpeg_command(
            ffmpeg_convert_command, f"{MP4_PATH} -> {RAW_WAV_PATH}"
        )

    @staticmethod
    def segment_wav():
        ZERO_SHOT_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
        for file_path in ZERO_SHOT_PROMPT_DIR.glob("*.wav"):
            file_path.unlink()
        audio = AudioSegment.from_wav(RAW_WAV_PATH)
        segments = silence.split_on_silence(
            audio,
            min_silence_len=MIN_SILENCE_LEN,
            silence_thresh=SILENCE_THRESH,
            keep_silence=KEEP_SILENCE,
        )

        if not segments:
            logger.warning(
                f"在 '{RAW_WAV_PATH.name}' 中未找到音频片段。请检查静音检测参数或音频内容。"
            )
            return

        for i, segment in enumerate(segments):
            segment_output_path = (
                ZERO_SHOT_PROMPT_DIR / f"{RAW_WAV_PATH.stem}_segment_{i:03d}.wav"
            )
            segment.export(str(segment_output_path), format="wav")
            logger.debug(f"已保存片段 {i} 到 {segment_output_path}")
        logger.info(
            f"成功将 '{RAW_WAV_PATH.name}' 分割成 {len(segments)} 个 WAV 文件到 {ZERO_SHOT_PROMPT_DIR}"
        )

    @staticmethod
    def epub2txt():
        if TXT_DIR.exists():
            for file_path in TXT_DIR.glob("*.txt"):
                file_path.unlink()

        book = epub.read_epub(EPUB_PATH)

        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

        for i, item in enumerate(items):
            soup = BeautifulSoup(item.get_content(), "html.parser")

            # Extract title for filename (clean from HTML content)
            title = None
            title_element = None

            # Try to find title in order of priority
            title_candidates = [
                soup.find("h1"),
                soup.find("h2"),
                soup.find("h3"),
                soup.find("title"),
            ]

            for candidate in title_candidates:
                if candidate and candidate.get_text().strip():
                    title = candidate.get_text().strip()
                    title_element = candidate
                    break

            # If no title found, use standard filename
            if not title:
                title = f"章节_{i + 1:03d}"

            # Clean title for filename (remove unwanted characters)
            safe_title = re.sub(r"[^\w\u4e00-\u9fff\s-]", "", title)
            safe_title = re.sub(r"\s+", "_", safe_title.strip())
            if not safe_title:
                safe_title = f"chapter_{i + 1:03d}"

            # Limit filename length and add sequential numbering
            numbered_title = f"{i + 1:03d}_{safe_title}"
            if len(numbered_title) > 50:
                # Keep the numbering, truncate the title part
                max_title_len = (
                    50 - 4
                )  # Reserve space for numbering (3 digits + underscore)
                numbered_title = f"{i + 1:03d}_{safe_title[:max_title_len]}"

            output_filename = f"{numbered_title}.txt"
            output_txt_path = TXT_DIR / output_filename

            # Remove script and style tags to clean the text content
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()

            # Remove the title element from content if found
            if title_element:
                title_element.decompose()

            footnote_selectors = [
                '[id*="footnote"]',
                '[id*="fn"]',
                '[id*="note"]',
                '[class*="footnote"]',
                '[class*="note"]',
                '[class*="endnote"]',
                '[epub|type="footnote"]',
                '[epub|type="note"]',
                '[epub|type="endnote"]',
                'aside[epub|type="footnote"]',
                'aside[class*="footnote"]',
                'div[class*="footnote"]',
                'section[class*="footnote"]',
            ]

            for selector in footnote_selectors:
                for footnote in soup.select(selector):
                    footnote.decompose()

            ref_selectors = [
                'a[href^="#"][href*="footnote"]',
                'a[href^="#"][href*="fn"]',
                'a[href^="#"][href*="note"]',
                'a[epub|type="noteref"]',
                'a[href*="note"]:not([href*="("])',
            ]

            for selector in ref_selectors:
                for ref in soup.select(selector):
                    ref.decompose()

            for sup in soup.find_all("sup"):
                sup_text = sup.get_text().strip()
                if re.match(r"^\d+$|[*†‡§¶]", sup_text):
                    sup.decompose()
            text = re.sub(r"\s+", "", soup.get_text()).strip()

            if len(text) < 20:
                logger.warning(
                    f"章节 {safe_title} 文本过短（{len(text)} 字符），跳过该章节音频生成。"
                )
                continue
            with open(output_txt_path, "w") as f:
                f.write(text)
            logger.success(f"已保存到 {output_txt_path}")

        logger.info("EPUB 到 TXT 转换完成。")

    def _initialize_cosyvoice(self):
        """
        初始化 CosyVoice2 模型并加载提示语音。
        延迟加载，仅在需要音频转换时才加载模型。
        """
        if self.cosyvoice is None:
            logger.info("正在初始化 CosyVoice2 模型...")
            self.cosyvoice = CosyVoice2(
                MODEL_PATH,  # 预训练模型路径
                load_jit=True,
                load_trt=True,
                load_vllm=True,
                fp16=True,
            )
        logger.success("CosyVoice2 模型初始化成功")

        self.prompt_speech_16k = load_wav(
            str(PROMPT_WAV_PATH), FFMPEG_AUDIO_SAMPLE_RATE
        )
        logger.success("提示语音加载成功")

    def txt2audio(self):
        self._initialize_cosyvoice()

        search_pattern = str(TXT_DIR / "*.txt")
        txt_files = sorted(glob.glob(search_pattern))

        for file_path in txt_files:
            file_path = Path(file_path)
            filename_stem = file_path.stem
            output_wav_subdir = AUDIO_DIR / filename_stem

            output_wav_subdir.mkdir(parents=True, exist_ok=True)
            for file in Path(output_wav_subdir).glob("*.wav"):
                file.unlink()
            for file in Path(output_wav_subdir).glob("*.txt"):
                file.unlink()

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            set_all_random_seed(42)  # 设置随机种子以确保输出一致

            for step, audio in enumerate(
                self.cosyvoice.inference_zero_shot(
                    text,
                    PROMPT_TEXT,
                    self.prompt_speech_16k,
                    stream=False,  # 设置为 True 以使用流式输出
                )
            ):
                output_wav_path = output_wav_subdir / f"{step:03d}.wav"
                output_txt_path = output_wav_subdir / f"{step:03d}.txt"

                with open(output_txt_path, "w") as f:
                    f.write(audio["text"])

                torchaudio.save(
                    str(output_wav_path),
                    audio["tts_speech"].cpu(),  # 如果在 GPU 上，将张量移到 CPU 再保存
                    self.cosyvoice.sample_rate,
                )

    @staticmethod
    def merge_chapter_audio(ext="mp3"):
        audio_sub_folders = sorted(glob.glob(str(AUDIO_DIR / "*/")))
        for chapter_folder in audio_sub_folders:
            chapter_folder = Path(chapter_folder)

            list_file_path = chapter_folder / "filelist.txt"
            output_mp3_path = AUDIO_DIR / f"{chapter_folder.stem}.{ext}"
            wav_files = sorted(glob.glob(str(chapter_folder / "*.wav")))
            with open(list_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join([f"file {wav_file}" for wav_file in wav_files]))

            if ext == "mp3":
                ffmpeg_command = [
                    "ffmpeg",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_file_path),
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    "-y",
                    str(output_mp3_path),
                ]
            elif ext == "wav":
                ffmpeg_command = [
                    "ffmpeg",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_file_path),
                    "-c",
                    "copy",
                    "-y",
                    str(output_mp3_path),
                ]
            else:
                raise ValueError("invalid ext")

            try:
                AudiobookConverter._run_ffmpeg_command(
                    ffmpeg_command, f"合并章节音频: {chapter_folder.stem}.{ext}"
                )
                os.remove(list_file_path)
            except subprocess.CalledProcessError:
                # 清理临时文件并继续处理下一个文件
                if list_file_path.exists():
                    os.remove(list_file_path)
                logger.error(f"跳过章节: {chapter_folder.stem} 由于ffmpeg命令失败")
                continue

    @staticmethod
    def generate_srt(chapter_folder: Path):
        def _seconds_to_srt_time(seconds):
            """将秒数转换为SRT格式的时间戳"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            milliseconds = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

        txt_files = sorted(chapter_folder.glob(("*.txt")))
        wav_files = sorted(chapter_folder.glob("*.wav"))
        chapter_srt_path = chapter_folder / f"{chapter_folder.stem}.srt"

        with open(chapter_srt_path, "w", encoding="utf-8") as srt_file:
            current_time = 0.0
            for i, (wav_file, txt_file) in enumerate(zip(wav_files, txt_files)):
                with open(txt_file, "r", encoding="utf-8") as txt_f:
                    text = txt_f.read().strip()

                if not text:
                    continue
                duration = len(AudioSegment.from_wav(wav_file)) / 1000.0

                start_time = current_time
                end_time = current_time + duration

                start_str = _seconds_to_srt_time(start_time)
                end_str = _seconds_to_srt_time(end_time)

                # 写入字幕条目
                srt_file.write(f"{i + 1}\n")
                srt_file.write(f"{start_str} --> {end_str}\n")
                srt_file.write(f"{text}\n\n")

                current_time = end_time

        return chapter_srt_path

    @staticmethod
    def merge_chapter_audio_with_subtitles():
        audio_sub_folders = sorted(glob.glob(str(AUDIO_DIR / "*/")))
        AudiobookConverter.merge_chapter_audio("wav")

        for chapter_folder in audio_sub_folders:
            chapter_folder = Path(chapter_folder)
            chapter_name = chapter_folder.stem

            chapter_mp4_path = AUDIO_DIR / f"{chapter_name}.mp4"
            chapter_audio_path = AUDIO_DIR / f"{chapter_name}.wav"

            chapter_srt_path = AudiobookConverter.generate_srt(chapter_folder)

            ffmpeg_video_command = [
                "ffmpeg",
                "-i",
                str(chapter_audio_path),  # 音频输入
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=1920x1080:r=1",  # 黑色视频背景
                "-vf",
                f"subtitles={chapter_srt_path}:force_style='Fontsize=24,PrimaryColour=&HFFFFFF'",  # 字幕
                "-c:a",
                "aac",
                "-b:a",
                "192k",  # 音频编码
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",  # 视频编码
                "-shortest",  # 匹配短的流长度
                "-y",
                str(chapter_mp4_path),
            ]

            try:
                AudiobookConverter._run_ffmpeg_command(
                    ffmpeg_video_command, f"生成带字幕视频: {chapter_name}.mp4"
                )
            except subprocess.CalledProcessError:
                logger.error(f"跳过章节: {chapter_name} 由于ffmpeg视频生成失败")
                continue


if __name__ == "__main__":
    converter = AudiobookConverter()
    # 制作zero shot
    # converter.mp42wav()
    # converter.segment_wav()
    # 制作电子书
    # converter.epub2txt()
    # converter.txt2audio()
    # converter.merge_chapter_audio()
    # 为每个章节合并音频并添加字幕
    converter.merge_chapter_audio_with_subtitles()
