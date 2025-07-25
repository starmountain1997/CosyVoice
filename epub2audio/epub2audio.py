import sys
import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
from loguru import logger
import torchaudio
import glob
from pathlib import Path
import subprocess

# Append path for Matcha-TTS and CosyVoice2
sys.path.append("third_party/Matcha-TTS")
# Ensure these imports are correct based on your project structure
# If vllm is not installed globally, you might need to adjust your environment or installation.
try:
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    from cosyvoice.utils.common import set_all_random_seed

    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
except ImportError as e:
    logger.error(
        f"Failed to import CosyVoice2 modules. Please ensure 'third_party/Matcha-TTS' is correctly set up and dependencies are installed. Error: {e}"
    )
    # Exit or handle gracefully if core components are missing
    sys.exit(1)

MODEL_PATH = "pretrained_models/CosyVoice2-0.5B"

# Define base paths for outputs and status file
BASE_DIR = Path(
    "/workspace"
)  # Adjust this base directory if your workspace is different
TXT_DIR = BASE_DIR / "epub_res"
AUDIO_DIR = BASE_DIR / "generated_audio"
STATUS_FILE = BASE_DIR / "conversion_status.json"
# New directory for zero-shot prompt segments
ZERO_SHOT_PROMPT_DIR = BASE_DIR / "zero_shot_prompts"

PROMPT_WAV_PATH = "./asset/ladynana.wav"
PROMPT_TEXT = "中国的兄弟姐妹们你们好，我是俄罗斯娜娜。昨天晚上我的评论区炸锅了。"


class AudiobookConverter:
    """
    A class to convert EPUB files into audiobooks with breakpoint recovery.

    The conversion process is divided into three logical stages:
    1. EPUB to TXT: Extracts text from EPUB chapters into individual text files.
    2. TXT to Audio: Generates WAV audio segments for each text file using CosyVoice2.
    3. Merge Audio: Concatenates WAV segments for each chapter into a single MP3 file.

    Breakpoint recovery is implemented by maintaining a status file (`conversion_status.json`)
    that tracks the completion status of each chapter at each stage.
    """

    def __init__(self, epub_path: str):
        """
        Initializes the AudiobookConverter.

        Args:
            epub_path (str): The path to the EPUB file to be converted.
            prompt_wav_path (str): The path to the WAV file used as a zero-shot prompt for CosyVoice2.
        """
        self.epub_path = Path(epub_path)
        if not self.epub_path.exists():
            raise FileNotFoundError(f"EPUB file not found at: {self.epub_path}")

        self.status_file_path = STATUS_FILE

        # Ensure output directories exist
        TXT_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        ZERO_SHOT_PROMPT_DIR.mkdir(
            parents=True, exist_ok=True
        )  # Ensure zero-shot prompt directory exists

        self.cosyvoice = None  # CosyVoice2 model will be initialized lazily
        self.prompt_speech_16k = None  # Prompt speech for CosyVoice2

    def epub2txt(self):
        """
        Converts an EPUB file into plain text files, one per chapter,
        and saves them in the TXT_DIR directory. This function supports breakpoint
        recovery by skipping chapters that have already been converted to text.
        自动移除脚注内容，包括引用链接和脚注本身。
        使用章节标题作为文件名，并在正文中移除标题。
        """
        logger.info(f"开始 EPUB 到 TXT 转换：{self.epub_path.name}...")
        
        # Clear TXT_DIR before generating new files
        if TXT_DIR.exists():
            logger.info(f"清理 TXT_DIR：{TXT_DIR}")
            try:
                for file_path in TXT_DIR.glob("*.txt"):
                    file_path.unlink()
                    logger.debug(f"已删除：{file_path}")
            except Exception as e:
                logger.error(f"清理 TXT_DIR 失败：{e}")
        try:
            book = epub.read_epub(self.epub_path)
        except Exception as e:
            logger.error(f"读取 EPUB 文件失败 {self.epub_path}：{e}")
            return

        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

        for i, item in enumerate(items):
            soup = BeautifulSoup(item.get_content(), "html.parser")

            # Extract title for filename (clean from HTML content)
            title = None
            title_element = None
            
            # Try to find title in order of priority
            title_candidates = [
                soup.find('h1'),
                soup.find('h2'),
                soup.find('h3'),
                soup.find('title'),
            ]
            
            for candidate in title_candidates:
                if candidate and candidate.get_text().strip():
                    title = candidate.get_text().strip()
                    title_element = candidate
                    break
            
            # If no title found, use standard filename
            if not title:
                title = f"章节_{i+1:03d}"
            
            # Clean title for filename (remove unwanted characters)
            safe_title = re.sub(r'[^\w\u4e00-\u9fff\s-]', '', title)
            safe_title = re.sub(r'\s+', '_', safe_title.strip())
            if not safe_title:
                safe_title = f"chapter_{i+1:03d}"
            
            # Limit filename length and add sequential numbering
            numbered_title = f"{i+1:03d}_{safe_title}"
            if len(numbered_title) > 50:
                # Keep the numbering, truncate the title part
                max_title_len = 50 - 4  # Reserve space for numbering (3 digits + underscore)
                numbered_title = f"{i+1:03d}_{safe_title[:max_title_len]}"
            
            output_filename = f"{numbered_title}.txt"
            output_txt_path = TXT_DIR / output_filename

            # Remove script and style tags to clean the text content
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()

            # Remove the title element from content if found
            if title_element:
                title_element.decompose()

            # Remove footnotes and references
            # 1. Remove common footnote containers
            footnote_selectors = [
                '[id*="footnote"]', '[id*="fn"]', '[id*="note"]',
                '[class*="footnote"]', '[class*="note"]', '[class*="endnote"]',
                '[epub|type="footnote"]', '[epub|type="note"]', '[epub|type="endnote"]',
                'aside[epub|type="footnote"]', 'aside[class*="footnote"]',
                'div[class*="footnote"]', 'section[class*="footnote"]',
            ]
            
            for selector in footnote_selectors:
                for footnote in soup.select(selector):
                    footnote.decompose()

            # 2. Remove footnote reference links with specific patterns
            ref_selectors = [
                'a[href^="#"][href*="footnote"]', 'a[href^="#"][href*="fn"]', 'a[href^="#"][href*="note"]',
                'a[epub|type="noteref"]', 'a[href*="note"]:not([href*="("])',
            ]
            
            for selector in ref_selectors:
                for ref in soup.select(selector):
                    ref.decompose()

            # 3. Remove superscript references that look like footnotes
            for sup in soup.find_all('sup'):
                # Remove superscripts containing only numbers or symbols (common footnote markers)
                sup_text = sup.get_text().strip()
                if re.match(r'^\d+$|[*†‡§¶]', sup_text):
                    sup.decompose()

            # Extract and clean text, removing extra whitespace and stripping
            text = re.sub(r"\s+", "", soup.get_text()).strip()

            if len(text) < 20:
                logger.warning(
                    f"章节 {safe_title} 文本过短（{len(text)} 字符），跳过该章节音频生成。"
                )
                continue
            with open(output_txt_path,"w") as f:
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
            logger.info("CosyVoice2 模型初始化成功")

            try:
                self.prompt_speech_16k = load_wav(str(PROMPT_WAV_PATH), 16000)
                logger.info("提示语音加载成功")
            except Exception as e:
                logger.error(
                    f"从 '{PROMPT_WAV_PATH}' 加载提示语音失败，请确保文件存在且为有效 WAV 格式。错误：{e}"
                )
                raise  # 重新抛出异常以停止执行

    def txt2audio_and_merge(self):
        try:
            self._initialize_cosyvoice()
        except Exception as e:
            logger.error(
                f"由于 CosyVoice2 初始化失败，跳过音频生成和合并，错误：{e}"
            )
            return

        search_pattern = str(TXT_DIR / "*.txt")
        txt_files = sorted(glob.glob(search_pattern))

        if not txt_files:
            raise FileNotFoundError(
                f"在 {TXT_DIR} 中未找到文本文件，无法生成音频。"
            )

        logger.info(f"找到 {len(txt_files)} 个文本文件，开始转换为音频并合并。")

        for file_path_str in txt_files:
            file_path = Path(file_path_str)
            filename_stem = file_path.stem  # 例如 "000"
            output_mp3_path = AUDIO_DIR / f"{filename_stem}.mp3"
            output_wav_subdir = (
                AUDIO_DIR / filename_stem
            )  # WAV 片段的子目录

            logger.info(
                f"处理章节 {filename_stem}：生成音频并合并为 MP3。"
            )

            # 如果 MP3 不存在或未标记为已生成，继续处理。
            # 清理 WAV 片段子目录以确保重新生成，
            # 特别是之前运行被中断时。
            if output_wav_subdir.exists():
                logger.info(
                    f"清理 {output_wav_subdir} 中现有的 WAV 文件以用于重新生成。"
                )
                try:
                    for existing_wav in output_wav_subdir.glob("*.wav"):
                        os.remove(existing_wav)
                except Exception as e:
                    logger.error(
                        f"清理 {output_wav_subdir} 中的 WAV 文件失败：{e}"
                    )
            output_wav_subdir.mkdir(
                parents=True, exist_ok=True
            )  # 确保目录存在

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                logger.warning(
                    f"文本文件 {file_path.name} 为空或仅包含空白字符，跳过音频生成。"
                )
                # 如果源文本为空，标记为已完成音频和合并
                continue

            logger.info(
                f"为 {file_path.name} 生成音频到目录 {output_wav_subdir}..."
            )
            set_all_random_seed(
                42
            )  # 设置随机种子以确保输出一致

            segments_generated = 0
            try:
                # CosyVoice2 推理生成音频片段
                for step, audio in enumerate(
                    self.cosyvoice.inference_zero_shot(
                        text,
                        PROMPT_TEXT,
                        self.prompt_speech_16k,
                        stream=False,  # 设置为 True 以使用流式输出
                    )
                ):
                    output_wav_path = output_wav_subdir / f"{step:03d}.wav"
                    torchaudio.save(
                        str(output_wav_path),
                        audio[
                            "tts_speech"
                        ].cpu(),  # 如果在 GPU 上，将张量移到 CPU 再保存
                        self.cosyvoice.sample_rate,
                    )
                    logger.debug(f"已保存片段 {step} 到 {output_wav_path}")
                    segments_generated += 1

                if segments_generated > 0:
                    logger.info(
                        f"已为 {file_path.name} 生成 {segments_generated} 个音频片段。"
                    )
                    # 继续合并生成的音频片段
                    self._merge_chapter_audio(
                        output_wav_subdir, output_mp3_path, filename_stem
                    )
                else:
                    logger.warning(
                        f"未为 {file_path.name} 生成任何音频片段，跳过合并。"
                    )

            except Exception as e:
                logger.error(f"为 {file_path.name} 生成音频时发生错误：{e}")

        logger.info("所有文本文件已处理完成音频生成和合并。")

    def _merge_chapter_audio(
        self, wav_subdir: Path, output_mp3_path: Path, chapter_filename_stem: str
    ):
        logger.info(
            f"开始合并章节 {chapter_filename_stem} 的音频，来源目录：{wav_subdir}..."
        )

        wav_files = sorted(wav_subdir.glob("*.wav"))

        if not wav_files:
            logger.warning(
                f"在 {wav_subdir} 中未找到 WAV 文件，跳过该章节合并。"
            )
            return

        # 为 FFmpeg 的 concat demuxer 创建临时文件列表
        list_file_path = wav_subdir / "filelist.txt"
        try:
            with open(list_file_path, "w", encoding="utf-8") as f:
                for wav_file in wav_files:
                    f.write(f"file '{wav_file.resolve()}'\n")
        except Exception as e:
            logger.error(f"在 {list_file_path} 创建 FFmpeg 文件列表失败：{e}")
            return

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

        try:
            logger.info(
                f"为章节 {chapter_filename_stem} 执行 FFmpeg 命令：{' '.join(ffmpeg_command)}"
            )
            result = subprocess.run(
                ffmpeg_command, check=True, capture_output=True, text=True
            )
            logger.info(
                f"成功合并章节 '{chapter_filename_stem}' 的音频到 {output_mp3_path}"
            )
        except Exception as e:
            logger.error(
                f"章节 '{chapter_filename_stem}' 执行 FFmpeg 时发生意外错误：{e}"
            )

        finally:
            if list_file_path.exists():
                try:
                    os.remove(list_file_path)
                except Exception as e:
                    logger.error(
                        f"删除临时文件列表 {list_file_path} 失败：{e}"
                    )
        logger.info(f"章节 {chapter_filename_stem} 的音频合并完成。")


if __name__ == "__main__":
    epub_file_to_convert = "/workspace/test.epub"

    try:
        converter = AudiobookConverter(epub_file_to_convert)
        # converter.epub2txt()
        converter.txt2audio_and_merge()

    except Exception as e:
        logger.critical(f"发生错误：{e}。")
