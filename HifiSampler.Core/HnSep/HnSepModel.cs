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
    private readonly InferenceSession? _session;

    public HnSepModel(string modelPath, string device, int deviceId, int nFft, int hopLength)
    {
        _nFft = nFft;
        _hopLength = hopLength;
        if (File.Exists(modelPath))
        {
            _session = OnnxUtils.CreateSession(modelPath, device, deviceId);
        }
    }

    public float[] PredictFromAudio(float[] audio)
    {
        if (_session is null || audio.Length == 0)
        {
            return audio.ToArray();
        }

        try
        {
            var window = StftEngine.BuildHannWindow(_nFft);
            var spec = StftEngine.Stft(audio, _nFft, _hopLength, _nFft, window, center: true);
            var f = spec.Bins;
            var t = spec.Frames;
            var fftSize = f * t;

            var inputArray = new float[2 * fftSize];
            Buffer.BlockCopy(spec.Real, 0, inputArray, 0, fftSize * sizeof(float));
            Buffer.BlockCopy(spec.Imaginary, 0, inputArray, fftSize * sizeof(float), fftSize * sizeof(float));
            var inputTensor = new DenseTensor<float>(inputArray, new[] { 1, 2, f, t });

            using var results = _session.Run([NamedOnnxValue.CreateFromTensor("input", inputTensor)]);
            var mask = results.First().AsTensor<float>(); // [1,2,f,t]

            var maskArray = mask.ToArray();
            if (maskArray.Length < inputArray.Length)
            {
                return audio.ToArray();
            }

            var maskReal = new float[fftSize];
            var maskImag = new float[fftSize];
            Buffer.BlockCopy(maskArray, 0, maskReal, 0, fftSize * sizeof(float));
            Buffer.BlockCopy(maskArray, fftSize * sizeof(float), maskImag, 0, fftSize * sizeof(float));

            var maskedReal = new float[fftSize];
            var maskedImag = new float[fftSize];
            ComplexMultiply(
                spec.Real,
                spec.Imaginary,
                maskReal,
                maskImag,
                maskedReal,
                maskedImag);

            var output = StftEngine.Istft(
                maskedReal,
                maskedImag,
                f,
                t,
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
    
    private static void ComplexMultiply(
        float[] inputReal,
        float[] inputImag,
        float[] maskReal,
        float[] maskImag,
        float[] outputReal,
        float[] outputImag)
    {
        var length = inputReal.Length;
        if (inputImag.Length != length ||
            maskReal.Length != length ||
            maskImag.Length != length ||
            outputReal.Length != length ||
            outputImag.Length != length)
        {
            throw new ArgumentException("All complex buffers must have the same length.");
        }

        var simd = Vector<float>.Count;
        var i = 0;
        for (; i <= length - simd; i += simd)
        {
            var ar = new Vector<float>(inputReal, i);
            var ai = new Vector<float>(inputImag, i);
            var br = new Vector<float>(maskReal, i);
            var bi = new Vector<float>(maskImag, i);
            (ar * br - ai * bi).CopyTo(outputReal, i);
            (ar * bi + ai * br).CopyTo(outputImag, i);
        }

        for (; i < length; i++)
        {
            var real = inputReal[i];
            var imag = inputImag[i];
            var maskR = maskReal[i];
            var maskI = maskImag[i];
            outputReal[i] = real * maskR - imag * maskI;
            outputImag[i] = real * maskI + imag * maskR;
        }
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
