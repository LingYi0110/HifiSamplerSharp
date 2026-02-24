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

    private readonly float[,] _slaneyMelBank = SlaneyMel.BuildMelFilterBank(nMels, nFft, sampleRate, fMin, fMax);
    private readonly float[] _window = StftEngine.BuildHannWindow(winLength);

    public float[,] Extract(float[] input, float keyShift = 0f, float speed = 1f)
    {
        if (input.Length == 0)
        {
            return new float[nMels, 0];
        }

        var factor = MathF.Pow(2f, keyShift / 12f);
        var hopLengthNew = (int)MathF.Round(hopLength * speed);

        var padLeft = (winLength - hopLengthNew) / 2;
        var padRight = (winLength - hopLengthNew + 1) / 2;
        var yPad = StftEngine.ReflectPad(input, padLeft, padRight);

        var spectrum = StftEngine.Stft(yPad, nFft, hopLengthNew, winLength, _window, center);
        var magnitudeMatrix = Magnitude(spectrum);
        var adjusted = AdjustBins(magnitudeMatrix, factor);
        return ApplyMelFilterBank(_slaneyMelBank, adjusted);
    }

    private static float[,] Magnitude(Complex[,] spectrum)
    {
        var bins = spectrum.GetLength(0);
        var frames = spectrum.GetLength(1);
        var output = new float[bins, frames];
        for (var bin = 0; bin < bins; bin++)
        {
            for (var frame = 0; frame < frames; frame++)
            {
                output[bin, frame] = (float)spectrum[bin, frame].Magnitude;
            }
        }

        return output;
    }

    private float[,] ApplyMelFilterBank(float[,] filterBank, float[,] input)
    {
        var nMels = filterBank.GetLength(0);
        var bins = filterBank.GetLength(1);
        if (input.GetLength(0) != bins)
        {
            throw new ArgumentException("Input bin count does not match mel filter bank.");
        }

        var frames = input.GetLength(1);
        var output = new float[nMels, frames];
        Parallel.For(0, nMels, mel =>
        {
            var start = -1;
            var endExclusive = 0;
            for (var bin = 0; bin < bins; bin++)
            {
                if (filterBank[mel, bin] <= 0f)
                {
                    continue;
                }

                if (start < 0)
                {
                    start = bin;
                }

                endExclusive = bin + 1;
            }

            if (start < 0)
            {
                return;
            }

            for (var bin = start; bin < endExclusive; bin++)
            {
                var weight = filterBank[mel, bin];
                if (weight == 0f)
                {
                    continue;
                }

                for (var frame = 0; frame < frames; frame++)
                {
                    output[mel, frame] += weight * input[bin, frame];
                }
            }
        });

        return output;
    }
    
    private float[,] AdjustBins(float[,] source, float factor)
    {
        var sourceBins = source.GetLength(0);
        var frames = source.GetLength(1);
        if (sourceBins == _targetBins && MathF.Abs(factor - 1f) <= 1e-6f)
        {
            return source;
        }

        var target = new float[_targetBins, frames];
        var sourceMaxBin = sourceBins - 1;
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
            if (blend <= 1e-6f || upperBin == lowerBin)
            {
                for (var frame = 0; frame < frames; frame++)
                {
                    target[targetBin, frame] = source[lowerBin, frame];
                }

                return;
            }

            for (var frame = 0; frame < frames; frame++)
            {
                var lower = source[lowerBin, frame];
                target[targetBin, frame] = lower + (source[upperBin, frame] - lower) * blend;
            }
        });

        return target;
    }
}
