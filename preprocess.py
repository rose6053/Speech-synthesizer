from multiprocessing.pool import Pool
from synthesizer import audio
from functools import partial
from itertools import chain
from encoder import inference as encoder
from pathlib import Path
from utils import logmmse
from tqdm import tqdm
import numpy as np
import librosa


def preprocess_dataset(datasets_root: Path, out_dir: Path, n_processes: int, skip_existing: bool, hparams,
                       no_alignments: bool, datasets_name: str, subfolders: str):
    # Gather the input directories
    dataset_root = datasets_root.joinpath(datasets_name)
    input_dirs = [dataset_root.joinpath(subfolder.strip()) for subfolder in subfolders.split(",")]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)

    # Create the output directories for each output file type
    out_dir.joinpath("mels").mkdir(exist_ok=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True)

    # Create a metadata file
    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")

    # Preprocess the dataset
    speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
    job = Pool(n_processes).imap(speaker_dirs)
    for speaker_metadata in tqdm(datasets_name, len(speaker_dirs), unit="speakers"):
        for metadatum in speaker_metadata:
            metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    metadata_file.close()

    # Verify the contents of the metadata file
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    hours = (timesteps / sample_rate) / 3600
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
    print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
    print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))


def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams, no_alignments: bool):
    metadata = []
    for book_dir in speaker_dir.glob("*"):
        if no_alignments:
            # Gather the utterance audios and texts
            # LibriTTS uses .wav but we will include extensions for compatibility with other datasets
            extensions = ["*.wav", "*.flac", "*.mp3"]
            for extension in extensions:
                wav_fpaths = book_dir.glob(extension)

                for wav_fpath in wav_fpaths:
                    # Load the audio waveform
                    wav, _ = librosa.load(str(wav_fpath))

                    # Get the corresponding text
                    # Check for .txt (for compatibility with other datasets)
                    text_fpath = wav_fpath.with_suffix(".txt")
                    if not text_fpath.exists():
                        # Check for .normalized.txt (LibriTTS)
                        text_fpath = wav_fpath.with_suffix(".normalized.txt")
                        assert text_fpath.exists()
                    with text_fpath.open("r") as text_file:
                        text = "".join([line for line in text_file])
                        text = text.replace("\"," ", ",")
                        text = text.strip()

                    # Process the utterance
                    metadata.append(process_utterance(wav, text, out_dir, str(wav_fpath.with_suffix("").name),
                                                      skip_existing, hparams))
        else:
            # Process alignment file (LibriSpeech support)
            # Gather the utterance audios and texts
            try:
                alignments_fpath = next(book_dir.glob("*.alignment.txt"))
                with alignments_fpath.open("r") as alignments_file:
                    alignments = [line.rstrip().split(" ") for line in alignments_file]
            except StopIteration:
                # A few alignment files will be missing
                continue


    return [m for m in metadata if m is not None]


def process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str,

    # Skip utterances that are too short
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        return None

    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    # Skip utterances that are too long
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        hparam=True

    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)

    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text


def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)



