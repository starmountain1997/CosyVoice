import soundfile as sf
import noisereduce as nr
import numpy as np
import os
import subprocess
from pathlib import Path
from loguru import logger
from pydub import AudioSegment, silence
from pathlib import Path
import glob

# --- 超参数配置 (Hyperparameter Configuration) ---

# FFmpeg 转换参数 (FFmpeg Conversion Parameters)
FFMPEG_AUDIO_SAMPLE_RATE = 16000  # 音频采样率 (Audio sample rate)
FFMPEG_AUDIO_CHANNELS = 1  # 音频通道数 (Number of audio channels)
FFMPEG_AUDIO_CODEC = "pcm_s16le"  # 音频编码器 (Audio codec)

# 音频分割参数 (Audio Segmentation Parameters)
MIN_SILENCE_LEN = 500  # 最小静音长度 (ms) (Minimum silence length in ms)
SILENCE_THRESH = -40  # 静音阈值 (dBFS) (Silence threshold in dBFS)
KEEP_SILENCE = 200  # 保留静音长度 (ms) (Silence to keep around segments in ms)

# 噪声消除参数 (Noise Reduction Parameters)
NOISE_REDUCE_PROP_DECREASE = 0.9  # 噪声减少比例 (Noise reduction proportion)
NOISE_ESTIMATION_DURATION = (
    0.2  # 噪声估算时长 (秒) (Duration for noise estimation in seconds)
)


def process_mp4_for_zero_shot_prompts(mp4_path: Path, output_dir: Path):
    if not mp4_path.exists():
        logger.error(f"MP4 文件未找到: {mp4_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_wav_path = output_dir / f"{mp4_path.stem}_temp.wav"

    logger.info(f"正在转换 MP4 文件 '{mp4_path.name}' 到 WAV: {temp_wav_path}...")
    # FFmpeg 命令将 MP4 转换为 WAV (16kHz, 单声道, 16-bit PCM 以兼容性)
    # FFmpeg command to convert MP4 to WAV (16kHz, mono, 16-bit PCM for compatibility)
    ffmpeg_convert_command = [
        "ffmpeg",
        "-i",
        str(mp4_path),
        "-vn",  # 无视频 (No video)
        "-acodec",
        FFMPEG_AUDIO_CODEC,  # PCM 16-bit little-endian
        "-ar",
        str(FFMPEG_AUDIO_SAMPLE_RATE),  # 16 kHz 采样率 (16 kHz sample rate)
        "-ac",
        str(FFMPEG_AUDIO_CHANNELS),  # 单声道音频 (Mono audio)
        "-y",  # 如果输出文件存在则覆盖 (Overwrite output file if it exists)
        str(temp_wav_path),
    ]

    try:
        subprocess.run(
            ffmpeg_convert_command, check=True, capture_output=True, text=True
        )
        logger.info(f"成功将 '{mp4_path.name}' 转换为 WAV.")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg MP4 到 WAV 转换失败 '{mp4_path.name}': {e}")
        logger.error(f"FFmpeg stdout:\n{e.stdout}")
        logger.error(f"FFmpeg stderr:\n{e.stderr}")
        return
    except FileNotFoundError:
        logger.error(
            "未找到 FFmpeg 命令 'ffmpeg'。请确保 FFmpeg 已安装并可在系统 PATH 中访问。"
            "FFmpeg command 'ffmpeg' not found. Please ensure FFmpeg is installed and accessible in your system's PATH."
        )
        return

    logger.info(f"正在将 WAV 文件 '{temp_wav_path.name}' 分割成句子...")
    try:
        audio = AudioSegment.from_wav(temp_wav_path)
        segments = silence.split_on_silence(
            audio,
            min_silence_len=MIN_SILENCE_LEN,
            silence_thresh=SILENCE_THRESH,
            keep_silence=KEEP_SILENCE,
        )

        if not segments:
            logger.warning(
                f"在 '{temp_wav_path.name}' 中未找到音频片段。请检查静音检测参数或音频内容。"
                f"No audio segments found in '{temp_wav_path.name}'. Check silence detection parameters or audio content."
            )
            return

        for i, segment in enumerate(segments):
            segment_output_path = output_dir / f"{mp4_path.stem}_segment_{i:03d}.wav"
            segment.export(str(segment_output_path), format="wav")
            logger.debug(f"已保存片段 {i} 到 {segment_output_path}")
        logger.info(
            f"成功将 '{temp_wav_path.name}' 分割成 {len(segments)} 个 WAV 文件到 {output_dir}."
        )

    except Exception as e:
        logger.error(f"WAV 分割过程中发生错误 '{temp_wav_path.name}': {e}")
    finally:
        # 清理临时 WAV 文件 (Clean up the temporary WAV file)
        if temp_wav_path.exists():
            try:
                os.remove(temp_wav_path)
                logger.debug(f"已删除临时 WAV 文件: {temp_wav_path}")
            except Exception as e:
                logger.error(
                    f"未能删除临时 WAV 文件 {temp_wav_path}: {e}"
                    f"Failed to remove temporary WAV file {temp_wav_path}: {e}"
                )

    logger.info("MP4 到分割 WAV 过程完成。")


# --- 使用示例 (Usage Example) ---
if __name__ == "__main__":
    mp4_path = Path("/workspace/ladynana.mp4")
    output_dir = Path("/workspace/ladynana")
    process_mp4_for_zero_shot_prompts(mp4_path, output_dir)
