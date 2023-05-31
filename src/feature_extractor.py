import librosa
import audio

class Feature_Extractor():
    def __init__(self, config) -> None:
        self.config = config
        
        self.STFT = audio.stft.TacotronSTFT(
            config["stft"]["filter_length"],
            config["stft"]["hop_length"],
            config["stft"]["win_length"],
            config["mel"]["n_mel_channels"],
            config["audio"]["sample_rate"],
            config["mel"]["mel_fmin"],
            config["mel"]["mel_fmax"],
        )
    
    def extract_mfcc(self, path):
        wav, _ = librosa.load(path, sr=self.config["audio"]["sample_rate"])
        mfcc = librosa.feature.mfcc(
            y=wav, 
            sr=self.config["audio"]["sample_rate"],
            hop_length=self.config["stft"]["hop_length"],
            win_length=self.config["stft"]["win_length"],
            n_mfcc=int(self.config["mfcc"]["n_mfcc"]),
            fmax=self.config["mfcc"]["mel_fmax"],
            fmin=self.config["mfcc"]["mel_fmin"])
        
        return mfcc
    
    def extract_mel_spectrogram(self, path, max_duration=None):
        wav, sr = librosa.load(path, sr=self.config["audio"]["sample_rate"])
        if max_duration is not None:
            wav = wav[0:int(sr*max_duration)]
        mel_spectrogram, _ = audio.tools.get_mel_from_wav(wav, self.STFT)
        
        return mel_spectrogram

    
if __name__ == "__main__":
    from yaml.loader import SafeLoader
    import yaml

    with open("configs/general_config.yml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)
    feature_extractor = Feature_Extractor(config)