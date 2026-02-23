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
    private readonly float[] _window = StftEngine.BuildHannWindow(winLength);

    public FloatMatrix Extract(float[] input, float keyShift = 0f, float speed = 1f)
    {
        if (input.Length == 0)
        {
            return new FloatMatrix(nMels, 0);
        }

        var factor = MathF.Pow(2f, keyShift / 12f);
        var hopLengthNew = (int)MathF.Round(hopLength * speed);

        var padLeft = (winLength - hopLengthNew) / 2;
        var padRight = (winLength - hopLengthNew + 1) / 2;
        var yPad = StftEngine.ReflectPad(input, padLeft, padRight);

        var spec = StftEngine.Stft(yPad, nFft, hopLengthNew, winLength, _window, center);
        var magnitude = Magnitude(spec.Real, spec.Imaginary);
        var magnitudeMatrix = FloatMatrix.FromFlat(spec.Bins, spec.Frames, magnitude, takeOwnership: true);
        var adjusted = AdjustBins(magnitudeMatrix, factor);
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
        float factor)
    {
        if (source.Rows == _targetBins && MathF.Abs(factor - 1f) <= 1e-6f)
        {
            return source;
        }

        var frames = source.Cols;
        var target = new FloatMatrix(_targetBins, frames);
        var sourceMaxBin = source.Rows - 1;
        Parallel.For(0, _targetBins, targetBin =>
        {
            var sourceBinPosition = targetBin / factor;
            if (sourceBinPosition < 0f || sourceBinPosition > sourceMaxBin)
            {
                return;
            }

            var lowerBin = Math.Clamp((int)sourceBinPosition, 0, sourceMaxBin);
            var upperBin = Math.Min(lowerBin + 1, sourceMaxBin);
            var blend = sourceBinPosition - lowerBin;
            var targetRow = target.RowSpan(targetBin);
            if (blend <= 1e-6f || upperBin == lowerBin)
            {
                source.RowSpan(lowerBin).CopyTo(targetRow);
                return;
            }

            var lowerRow = source.RowSpan(lowerBin);
            var upperRow = source.RowSpan(upperBin);
            for (var frame = 0; frame < frames; frame++)
            {
                targetRow[frame] = lowerRow[frame] + (upperRow[frame] - lowerRow[frame]) * blend;
            }
        });

        return target;
    }
}
