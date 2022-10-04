import torchaudio
import torch, torch.nn as nn

class Wav2Vec(nn.Module):
    def __init__(self, device = None) -> None:
        super().__init__()
        
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        # self.AUDIO_MAX_LENGTH = max_audio_len
        self.model = self.bundle.get_model()
        if device:
            self.model.to(device)

    def forward(self, waveform):
        '''
        support batch operation
        get last layer (#12) features
        '''
        return self.model.extract_features(waveform)[0][-1]

    class GreedyCTCDecoder(nn.Module):
        def __init__(self, labels, blank=0):
            super().__init__()
            self.labels = labels
            self.blank = blank

        def forward(self, emission: torch.Tensor) -> str:
            """Given a sequence emission over labels, get the best path string
            Args:
            emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

            Returns:
            str: The resulting transcript
            """
            indices = torch.argmax(emission, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank]
            return "".join([self.labels[i] for i in indices])
    
    def inference(self, waveform):
        with torch.inference_mode():
            emission, _ = self.model(waveform[:,:self.AUDIO_MAX_LENGTH])
        decoder = Wav2Vec.GreedyCTCDecoder(labels=self.bundle.get_labels())
        transcript = decoder(emission[0])
        return transcript