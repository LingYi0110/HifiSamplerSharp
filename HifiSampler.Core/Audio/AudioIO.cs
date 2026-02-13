using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace HifiSampler.Core.Audio;

public class AudioIO
{
    public static float[] ReadMono(string path, out int sampleRate, int? targetSampleRate = null)
    {
        if (String.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Audio file path cannot be null or empty.");
        }
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Audio file not found: {path}");
        }
        
        try
        {
            using var reader = new AudioFileReader(path);
            var originalSampleRate = reader.WaveFormat.SampleRate;
            
            ISampleProvider sampleProvider = reader;
            
            // Resample
            if (targetSampleRate.HasValue && targetSampleRate.Value > 0 && targetSampleRate.Value != originalSampleRate)
            {
                var resampler = new WdlResamplingSampleProvider(sampleProvider, targetSampleRate.Value);
                sampleProvider = resampler;
                sampleRate = targetSampleRate.Value;
            }
            else
            {
                sampleRate = originalSampleRate;
            }
            
            // Mix to mono
            if (sampleProvider.WaveFormat.Channels > 1)
            {
                sampleProvider = sampleProvider.ToMono();
            }

            // Read all sample and convert to float array
            var buffer = new List<float>();
            var readBuffer = new float[sampleProvider.WaveFormat.SampleRate * sampleProvider.WaveFormat.Channels];
            int samplesRead;
            
            while ((samplesRead = sampleProvider.Read(readBuffer, 0, readBuffer.Length)) > 0)
            {
                buffer.AddRange(readBuffer.Take(samplesRead));
            }

            return buffer.ToArray();
        }
        catch (Exception ex)
        {
            throw new InvalidDataException($"Error reading audio file: {path}", ex);
        }
    }

    public static void WriteWavMono(string path, float[] data, int sampleRate)
    {
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var waveFormat = new WaveFormat(sampleRate, 16, 1);
        using var writer = new WaveFileWriter(path, waveFormat);
        var pcm = new byte[data.Length * 2];
        for (var i = 0; i < data.Length; i++)
        {
            var sample = (short)Math.Clamp(data[i] * short.MaxValue, short.MinValue, short.MaxValue);
            pcm[i * 2] = (byte)(sample & 0xff);
            pcm[i * 2 + 1] = (byte)((sample >> 8) & 0xff);
        }

        writer.Write(pcm, 0, pcm.Length);
    }
}
