import soundfile as sf

data, samplerate = sf.read("./LizNelson_Rainfall_MIX.wav")

print(data.size)