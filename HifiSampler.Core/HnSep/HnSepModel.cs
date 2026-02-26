using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using HifiSampler.Core.Stft;
using HifiSampler.Core.Utils;

namespace HifiSampler.Core.HnSep;

public sealed class HnSepModel : IHnSep, IDisposable
{
    private readonly int _nFft;
    private readonly int _hopLength;
    private readonly InferenceSession _session;

    public HnSepModel(string modelPath, string device, int deviceId, int nFft, int hopLength)
    {
        _nFft = nFft;
        _hopLength = hopLength;
        _session = OnnxUtils.CreateSession(modelPath, device, deviceId);
    }

    public float[] PredictFromAudio(float[] audio)
    {
        if (audio.Length == 0)
        {
            return audio.ToArray();
        }

        try
        {
            var window = StftEngine.BuildHannWindow(_nFft);
            var spectrum = StftEngine.Stft(audio, _nFft, _hopLength, _nFft, window, center: true);
            var f = spectrum.GetLength(0);
            var t = spectrum.GetLength(1);
            var fftSize = f * t;

            var inputArray = new float[2 * fftSize];
            var offset = 0;
            for (var bin = 0; bin < f; bin++)
            {
                for (var frame = 0; frame < t; frame++)
                {
                    inputArray[offset++] = (float)spectrum[bin, frame].Real;
                }
            }

            for (var bin = 0; bin < f; bin++)
            {
                for (var frame = 0; frame < t; frame++)
                {
                    inputArray[offset++] = (float)spectrum[bin, frame].Imaginary;
                }
            }

            var inputTensor = new DenseTensor<float>(inputArray, new[] { 1, 2, f, t });

            using var results = _session.Run([NamedOnnxValue.CreateFromTensor("input", inputTensor)]);
            var mask = results.First().AsTensor<float>(); // [1,2,f,t]

            var maskArray = mask.ToArray();
            if (maskArray.Length < inputArray.Length)
            {
                return audio.ToArray();
            }

            var maskedSpectrum = new Complex[f, t];
            for (var bin = 0; bin < f; bin++)
            {
                for (var frame = 0; frame < t; frame++)
                {
                    var idx = bin * t + frame;
                    var maskComplex = new Complex(maskArray[idx], maskArray[fftSize + idx]);
                    maskedSpectrum[bin, frame] = spectrum[bin, frame] * maskComplex;
                }
            }

            var output = StftEngine.Istft(
                maskedSpectrum,
                _nFft,
                _hopLength,
                _nFft,
                window,
                center: true,
                expectedLength: audio.Length);

            if (output.Length == audio.Length)
            {
                return output;
            }

            var result = new float[audio.Length];
            Array.Copy(output, 0, result, 0, Math.Min(output.Length, result.Length));
            return result;
        }
        catch
        {
            return audio.ToArray();
        }
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}
