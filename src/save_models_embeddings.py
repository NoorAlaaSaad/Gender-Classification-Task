import os
import torch
import numpy as np
import time
import json

from pprint import pprint
from tqdm import tqdm

# -----------------------
#  Additional model imports
# -----------------------
# Silero
silero_model, silero_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, read_audio, _, _) = silero_utils

# Wav2Vec2
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Resemblyzer
from resemblyzer import VoiceEncoder, preprocess_wav

# ECAPA
from speechbrain.inference import EncoderClassifier


class MultiModelVoiceEmbedder:
    def __init__(self, models_dir="models-embeddings"):
        """
        Load Silero VAD, Wav2Vec2, Resemblyzer, and ECAPA-TDNN into memory.
        The idea is to have everything in one place so we can choose 
        which embedding method to call at any time.
        """
        # Create the output folder if it doesn't exist
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        # -------------- Silero VAD (CPU) --------------
        self.silero_model = silero_model
        self.silero_model_device = next(self.silero_model.parameters()).device
        print(f"Silero VAD loaded on {self.silero_model_device}")
        print("*" * 50)

        # -------------- Wav2Vec2 (GPU if available) --------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        wav2vec_model_name = "facebook/wav2vec2-base-960h"
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_name)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        self.wav2vec_model.to(self.device)
        self.wav2vec_model.eval()
        print(f"Wav2Vec2 model '{wav2vec_model_name}' loaded on {self.device}")
        print("*" * 50)

        # -------------- Resemblyzer (GPU if available) --------------
        self.resemblyzer_encoder = VoiceEncoder(device=self.device)
        print(f"Resemblyzer loaded on {self.resemblyzer_encoder.device}")
        print("*" * 50)

        # -------------- ECAPA-TDNN (GPU if available) --------------
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        self.ecapa.eval()
        print(f"ECAPA-TDNN loaded on {self.device}")
        print("*" * 50)

    def _extract_voiced_portions(self, audio_path, sample_rate=16000):
        """
        1) Read the audio from `audio_path`.
        2) Perform VAD to retrieve speech timestamps.
        3) Concatenate all speech segments into one NumPy array (float32).
        Returns:
            merged_voiced (np.ndarray) or None if no speech is found.
        """
        # Read audio at 16k
        audio_tensor = read_audio(audio_path, sampling_rate=sample_rate)

        # Run VAD
        speech_segments = get_speech_timestamps(
            audio_tensor,
            self.silero_model,
            sampling_rate=sample_rate,
            return_seconds=True,
            visualize_probs=False
        )
        if not speech_segments:
            return None

        audio_np = audio_tensor.numpy()
        voiced_parts = []
        for seg in speech_segments:
            start_sec, end_sec = seg["start"], seg["end"]
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            voiced_parts.append(audio_np[start_sample:end_sample])

        merged_voiced = np.concatenate(voiced_parts, axis=0).astype("float32")
        return merged_voiced

    # -------------------- WAV2VEC2 --------------------
    def compute_file_embedding_wav2vec(self, audio_path):
        """
        Compute a single Wav2Vec2 embedding for the voiced portions of `audio_path`.
        """
        merged_voiced = self._extract_voiced_portions(audio_path)
        if merged_voiced is None:
            return None

        inputs = self.wav2vec_feature_extractor(
            merged_voiced,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_values = inputs["input_values"].to(self.device)

        with torch.no_grad():
            outputs = self.wav2vec_model(input_values)
            hidden_states = outputs.last_hidden_state
            # Mean-pool across time to get a single vector
            embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()

        return embedding

    def main_wav2vec(self, male_folder, female_folder, output_filename="wav2vec-embeddings.jsonl"):
        """
        Loop over files in male/female folders, compute Wav2Vec2 embeddings, and save to JSONL.
        """
        output_path = os.path.join(self.models_dir, output_filename)
        start_time = time.time()
        with open(output_path, "w") as f_out:
            for folder_path, label in [(male_folder, "M"), (female_folder, "F")]:
                files_m4a = [f for f in os.listdir(folder_path) if f.lower().endswith(".m4a")]
                for fname in tqdm(files_m4a, desc=f"Wav2Vec2 -> {label}", unit="files"):
                    fpath = os.path.join(folder_path, fname)
                    if not os.path.isfile(fpath):
                        continue

                    emb = self.compute_file_embedding_wav2vec(fpath)
                    if emb is not None:
                        record = {
                            "filepath": fpath,
                            "label": label,
                            "embedding": emb.tolist()
                        }
                        f_out.write(json.dumps(record) + "\n")

        end_time = time.time()
        print(f"[Wav2Vec2] Saved embeddings to {output_path}")
        print(f"[Wav2Vec2] Execution time: {end_time - start_time:.2f} seconds")

    # -------------------- RESEMBLYZER --------------------
    def compute_file_embedding_resemblyzer(self, audio_path):
        """
        Compute a single Resemblyzer embedding for the voiced portions of `audio_path`.
        """
        merged_voiced = self._extract_voiced_portions(audio_path)
        if merged_voiced is None:
            return None

        # Resemblyzer approach: pass merged voiced array into preprocess_wav
        # which does internal normalization / resampling
        merged_audio_preproc = preprocess_wav(merged_voiced, source_sr=16000)

        # Now compute the embedding
        embedding = self.resemblyzer_encoder.embed_utterance(merged_audio_preproc)
        return embedding

    def main_resemblyzer(self, male_folder, female_folder, output_filename="resemblyzer-embeddings.jsonl"):
        """
        Loop over files in male/female folders, compute Resemblyzer embeddings, and save to JSONL.
        """
        output_path = os.path.join(self.models_dir, output_filename)
        start_time = time.time()
        with open(output_path, "w") as f_out:
            for folder_path, label in [(male_folder, "M"), (female_folder, "F")]:
                files_m4a = [f for f in os.listdir(folder_path) if f.lower().endswith(".m4a")]
                for fname in tqdm(files_m4a, desc=f"Resemblyzer -> {label}", unit="files"):
                    fpath = os.path.join(folder_path, fname)
                    if not os.path.isfile(fpath):
                        continue

                    emb = self.compute_file_embedding_resemblyzer(fpath)
                    if emb is not None:
                        record = {
                            "filepath": fpath,
                            "label": label,
                            "embedding": emb.tolist()
                        }
                        f_out.write(json.dumps(record) + "\n")

        end_time = time.time()
        print(f"[Resemblyzer] Saved embeddings to {output_path}")
        print(f"[Resemblyzer] Execution time: {end_time - start_time:.2f} seconds")

    # -------------------- ECAPA --------------------
    def compute_file_embedding_ecapa(self, audio_path):
        """
        Compute a single ECAPA-TDNN embedding for the voiced portions of `audio_path`.
        """
        merged_voiced = self._extract_voiced_portions(audio_path)
        if merged_voiced is None:
            return None

        # Convert merged_voiced to Torch tensor
        waveform = torch.from_numpy(merged_voiced).unsqueeze(0).to(self.device)

        # Encode
        with torch.no_grad():
            embedding = self.ecapa.encode_batch(waveform)  # [batch, 1, emb_dim]
            embedding = embedding.squeeze().cpu().numpy()

        return embedding

    def main_ecapa(self, male_folder, female_folder, output_filename="ecapa-embeddings.jsonl"):
        """
        Loop over files in male/female folders, compute ECAPA-TDNN embeddings, and save to JSONL.
        """
        output_path = os.path.join(self.models_dir, output_filename)
        start_time = time.time()
        with open(output_path, "w") as f_out:
            for folder_path, label in [(male_folder, "M"), (female_folder, "F")]:
                files_m4a = [f for f in os.listdir(folder_path) if f.lower().endswith(".m4a")]
                for fname in tqdm(files_m4a, desc=f"ECAPA -> {label}", unit="files"):
                    fpath = os.path.join(folder_path, fname)
                    if not os.path.isfile(fpath):
                        continue

                    emb = self.compute_file_embedding_ecapa(fpath)
                    if emb is not None:
                        record = {
                            "filepath": fpath,
                            "label": label,
                            "embedding": emb.tolist()
                        }
                        f_out.write(json.dumps(record) + "\n")

        end_time = time.time()
        print(f"[ECAPA] Saved embeddings to {output_path}")
        print(f"[ECAPA] Execution time: {end_time - start_time:.2f} seconds")


# --------------------
# Example usage (uncomment and set your paths)
# --------------------
if __name__ == "__main__":
    embedder = MultiModelVoiceEmbedder()
    male_folder_path = "/home/noor-alaa/Documents/Cyshield/data/VoxCeleb_gender/males"
    female_folder_path = "/home/noor-alaa/Documents/Cyshield/data/VoxCeleb_gender/females"

    # Wav2Vec2
    embedder.main_wav2vec(male_folder_path, female_folder_path)

    # Resemblyzer
    embedder.main_resemblyzer(male_folder_path, female_folder_path)

    # ECAPA
    embedder.main_ecapa(male_folder_path, female_folder_path)
