using System.Collections.Concurrent;
using System.Numerics;
using HifiSampler.Core.Stft;

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
    private readonly int _targetBins = nFft / 2 + 1;

    private readonly FloatMatrix _melBank = SlaneyMel.BuildMelFilterBank(nMels, nFft, sampleRate, fMin, fMax);
    private readonly ConcurrentDictionary<int, float[]> _hannWindowCache = new();

    public FloatMatrix Extract(float[] input, float keyShift = 0f, float speed = 1f)
    {
        if (input.Length == 0)
        {
            return new FloatMatrix(nMels, 0);
        }

        var factor = MathF.Pow(2f, keyShift / 12f);
        var nFftNew = (int)MathF.Round(nFft * factor);
        var winLengthNew = (int)MathF.Round(winLength * factor);
        var hopLengthNew = (int)MathF.Round(hopLength * speed);

        var padLeft = (winLengthNew - hopLengthNew) / 2;
        var padRight = (winLengthNew - hopLengthNew + 1) / 2;
        var yPad = StftEngine.ReflectPad(input, padLeft, padRight);
        var window = _hannWindowCache.GetOrAdd(winLengthNew, StftEngine.BuildHannWindow);

        var spec = StftEngine.Stft(yPad, nFftNew, hopLengthNew, winLengthNew, window, center);
        var magnitude = Magnitude(spec.Real, spec.Imaginary);
        var magnitudeMatrix = FloatMatrix.FromFlat(spec.Bins, spec.Frames, magnitude, takeOwnership: true);
        var adjusted = AdjustBins(magnitudeMatrix, winLengthNew, keyShift);
        return FloatMatrix.Multiply(_melBank, adjusted, parallel: true);
    }
    private static float[] Magnitude(float[] real, float[] imaginary)
    {
        var length = Math.Min(real.Length, imaginary.Length);
        var output = new float[length];

        var simd = Vector<float>.Count;
        var i = 0;
        for (; i <= length - simd; i += simd)
        {
            var r = new Vector<float>(real, i);
            var im = new Vector<float>(imaginary, i);
            Vector.SquareRoot(r * r + im * im).CopyTo(output, i);
        }

        for (; i < length; i++)
        {
            output[i] = MathF.Sqrt(real[i] * real[i] + imaginary[i] * imaginary[i]);
        }

        return output;
    }
    private FloatMatrix AdjustBins(
        FloatMatrix source,
        int winLengthNew,
        float keyShift)
    {
        if (source.Rows == _targetBins && MathF.Abs(keyShift) <= 1e-6f)
        {
            return source;
        }

        var frames = source.Cols;
        var target = new FloatMatrix(_targetBins, frames);
        var copyBins = Math.Min(source.Rows, _targetBins);
        Parallel.For(0, copyBins, bin =>
        {
            source.RowSpan(bin).CopyTo(target.RowSpan(bin));
        });

        if (MathF.Abs(keyShift) > 1e-6f)
        {
            target.ScaleInPlace((float)winLength / winLengthNew, parallel: true);
        }

        return target;
    }
}
