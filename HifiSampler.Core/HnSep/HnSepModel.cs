using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

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
            _session = Utils.OnnxUtils.CreateSession(modelPath, device, deviceId);
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
            using var wav = tensor(audio, dtype: ScalarType.Float32).unsqueeze(0);
            using var window = hann_window(_nFft, dtype: ScalarType.Float32);
            using var spec = stft(
                input: wav,
                n_fft: _nFft,
                hop_length: _hopLength,
                win_length: _nFft,
                window: window,
                center: true,
                pad_mode: PaddingModes.Reflect,
                normalized: false,
                onesided: true,
                return_complex: false);

            var shape = spec.shape;
            var f = (int)shape[1];
            var t = (int)shape[2];
            var c = (int)shape[3];
            if (c != 2)
            {
                return audio.ToArray();
            }

            // [1, f, t, 2] -> [1, 2, f, t]
            using var specInput = spec.permute(0, 3, 1, 2).contiguous().cpu();
            var inputArray = specInput.data<float>().ToArray();
            var inputTensor = new DenseTensor<float>(inputArray, new[] { 1, 2, f, t });

            using var results = _session.Run([NamedOnnxValue.CreateFromTensor("input", inputTensor)]);
            var mask = results.First().AsTensor<float>(); // [1,2,f,t]

            var masked = new DenseTensor<float>(new[] { 1, f, t, 2 });
            for (var fi = 0; fi < f; fi++)
            {
                for (var ti = 0; ti < t; ti++)
                {
                    var real = inputTensor[0, 0, fi, ti];
                    var imag = inputTensor[0, 1, fi, ti];
                    var mReal = mask[0, 0, fi, ti];
                    var mImag = mask[0, 1, fi, ti];
                    masked[0, fi, ti, 0] = real * mReal - imag * mImag;
                    masked[0, fi, ti, 1] = real * mImag + imag * mReal;
                }
            }

            using var specTensor = tensor(masked.Buffer.ToArray(), new long[] { 1, f, t, 2 }, dtype: ScalarType.Float32);
            using var complex = torch.view_as_complex(specTensor);
            using var wavPred = istft(
                input: complex,
                n_fft: _nFft,
                hop_length: _hopLength,
                win_length: _nFft,
                window: window,
                center: true);

            var output = wavPred.squeeze(0).cpu().data<float>().ToArray();
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
        _session?.Dispose();
    }
}
