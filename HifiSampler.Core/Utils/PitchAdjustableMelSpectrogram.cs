using System.Collections.Concurrent;
using NWaves.Filters.Fda;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace HifiSampler.Core.Utils;

public sealed class PitchAdjustableMelSpectrogram(
    int sampleRate = 44100,
    int nFft = 2048,
    int winLength = 2048,
    int hopLength = 512,
    float fMin = 40f,
    float fMax = 16000f,
    int nMels = 128,
    bool center = false)
{
    private readonly Tensor _melTensor =
        InitializeMelTensor(FilterBanks.MelBankSlaney(nMels, nFft, sampleRate, fMin, fMax));

    private readonly ConcurrentDictionary<string, Tensor> _hannWindowCache = new();

    private static Tensor InitializeMelTensor(float[][] melBank)
    {
        var rows = melBank.Length;
        var cols = melBank[0].Length;
        var flat = new float[rows * cols];
        for (var i = 0; i < rows; i++)
        {
            Array.Copy(melBank[i], 0, flat, i * cols, cols);
        }

        return tensor(flat, dtype: ScalarType.Float32).reshape(rows, cols);
    }

    public float[,] Extract(float[] input, float keyShift = 0f, float speed = 1f)
    {
        using var _ = torch.no_grad();

        var factor = MathF.Pow(2f, keyShift / 12f);
        var nFftNew = Math.Max(16, (int)MathF.Round(nFft * factor));
        var winLengthNew = Math.Max(16, (int)MathF.Round(winLength * factor));
        var hopLengthNew = Math.Max(1, (int)MathF.Round(hopLength * speed));

        using var y = tensor(input, dtype: ScalarType.Float32).unsqueeze(0);
        var padLeft = (winLengthNew - hopLengthNew) / 2;
        var padRight = (winLengthNew - hopLengthNew + 1) / 2;
        using var yPad = functional.pad(y.unsqueeze(1), (padLeft, padRight), PaddingModes.Reflect).squeeze(1);

        var key = $"{keyShift:F3}";
        if (!_hannWindowCache.TryGetValue(key, out var hann))
        {
            hann = torch.hann_window(winLengthNew, dtype: ScalarType.Float32);
            _hannWindowCache[key] = hann;
        }

        using var spec0 = torch.stft(
            yPad,
            n_fft: nFftNew,
            hop_length: hopLengthNew,
            win_length: winLengthNew,
            window: hann,
            center: center,
            pad_mode: PaddingModes.Reflect,
            normalized: false,
            onesided: true,
            return_complex: true).abs();

        var spec = spec0;
        if (Math.Abs(keyShift) > 1e-6)
        {
            var size = nFft / 2 + 1;
            var resize = (int)spec.shape[1];
            if (resize < size)
            {
                spec = functional.pad(spec, (0, 0, 0, size - resize));
            }

            var idx = torch.arange(size, dtype: ScalarType.Int64);
            spec = spec.index_select(1, idx);
            spec *= (float)winLength / winLengthNew;
        }

        using var mel = torch.matmul(_melTensor, spec).squeeze(0);
        return TensorTo2D(mel);
    }

    private static float[,] TensorTo2D(Tensor tensor2d)
    {
        var rows = (int)tensor2d.shape[0];
        var cols = (int)tensor2d.shape[1];
        var flat = tensor2d.contiguous().cpu().data<float>().ToArray();
        var result = new float[rows, cols];
        var idx = 0;
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                result[r, c] = flat[idx++];
            }
        }

        return result;
    }
}
