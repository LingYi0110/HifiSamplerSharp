using System.Collections.Concurrent;
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
    private readonly int _nFft = nFft;
    private readonly int _winLength = winLength;
    private readonly int _hopLength = hopLength;
    private readonly int _nMels = nMels;
    private readonly bool _center = center;
    private readonly int _targetBins = nFft / 2 + 1;

    private readonly float[] _melBank = SlaneyMel.BuildMelFilterBank(nMels, nFft, sampleRate, fMin, fMax);
    private readonly ConcurrentDictionary<int, float[]> _hannWindowCache = new();

    public float[,] Extract(float[] input, float keyShift = 0f, float speed = 1f)
    {
        if (input.Length == 0)
        {
            return new float[_nMels, 0];
        }

        var factor = MathF.Pow(2f, keyShift / 12f);
        var nFftNew = (int)MathF.Round(_nFft * factor);
        var winLengthNew = (int)MathF.Round(_winLength * factor);
        var hopLengthNew = (int)MathF.Round(_hopLength * speed);

        var padLeft = (winLengthNew - hopLengthNew) / 2;
        var padRight = (winLengthNew - hopLengthNew + 1) / 2;
        var yPad = StftEngine.ReflectPad(input, padLeft, padRight);
        var window = _hannWindowCache.GetOrAdd(winLengthNew, StftEngine.BuildHannWindow);

        var spec = StftEngine.Stft(yPad, nFftNew, hopLengthNew, winLengthNew, window, _center);
        var magnitude = SlaneyMel.Magnitude(spec.Real, spec.Imaginary);
        var adjusted = AdjustBins(magnitude, spec.Bins, spec.Frames, winLengthNew, keyShift);
        return SlaneyMel.Multiply(_melBank, _nMels, _targetBins, adjusted, spec.Frames);
    }

    private float[] AdjustBins(
        float[] source,
        int sourceBins,
        int frames,
        int winLengthNew,
        float keyShift)
    {
        if (sourceBins == _targetBins && MathF.Abs(keyShift) <= 1e-6f)
        {
            return source;
        }

        var target = new float[_targetBins * frames];
        var copyBins = Math.Min(sourceBins, _targetBins);
        Parallel.For(0, copyBins, bin =>
        {
            Array.Copy(source, bin * frames, target, bin * frames, frames);
        });

        if (MathF.Abs(keyShift) > 1e-6f)
        {
            SlaneyMel.ScaleInPlace(target, (float)_winLength / winLengthNew);
        }

        return target;
    }
}
