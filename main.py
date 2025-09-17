import sounddevice as sd
from datasets import Audio, load_dataset
from transformers import AutoProcessor, EncodecModel

BANDWIDTH = 6  # kbps, try 6, 12, or 24. Higher bandwidth = better quality.

# dummy dataset, however you can swap this with an dataset on the 🤗 hub or bring your own
librispeech_dummy = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz", use_fast=True)

# cast the audio data to the correct sampling rate for the model
librispeech_dummy = librispeech_dummy.cast_column(
    "audio", Audio(sampling_rate=processor.sampling_rate)
)
audio_sample = librispeech_dummy[0]["audio"]["array"]

# pre-process the inputs
inputs = processor(
    raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt"
)

print("Playing original audio ...")
sd.play(inputs["input_values"].squeeze().numpy(), samplerate=processor.sampling_rate)
sd.wait()

# explicitly encode then decode the audio inputs
encoder_outputs = model.encode(
    inputs["input_values"], inputs["padding_mask"], bandwidth=BANDWIDTH
)
audio_values = model.decode(
    encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"]
)[0]

print("Playing reconstructed audio ...")
sd.play(audio_values.detach().squeeze().numpy(), samplerate=processor.sampling_rate)
sd.wait()

print(f"""
Bandwith:              {BANDWIDTH} kbps
Input values shape:    {inputs["input_values"].shape} (batch_size, channels, samples)
Encoder outputs shape: {encoder_outputs["audio_codes"].shape} (batch_size, channels, number_of_codebooks, frames)
Reconstructed shape:   {audio_values.shape} (batch_size, channels, samples)
""")

# (Optional) or the equivalent with a forward pass
# audio_values = model(
#     inputs["input_values"], inputs["padding_mask"], bandwidth=BANDWIDTH
# ).audio_values

# (Optional) you can also extract the discrete codebook representation for LM
# tasks output: concatenated tensor of all the representations
# audio_codes = model(
#     inputs["input_values"], inputs["padding_mask"], bandwidth=BANDWIDTH
# ).audio_codes
