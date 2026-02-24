using System.Numerics;

namespace HifiSampler.Core.Utils;

internal static class SlaneyMel
{
    public static float[,] BuildMelFilterBank(
        int nMels,
        int nFft,
        int sampleRate,
        float fMin,
        float fMax)
    {
        var bins = nFft / 2 + 1;
        var filters = new float[nMels, bins];

        var clampedFMin = Math.Clamp(fMin, 0f, sampleRate / 2f);
        var clampedFMax = Math.Clamp(fMax, clampedFMin + 1f, sampleRate / 2f);

        var melMin = HzToMelSlaney(clampedFMin);
        var melMax = HzToMelSlaney(clampedFMax);
        var melPoints = new float[nMels + 2];
        var melStep = (melMax - melMin) / (nMels + 1);
        for (var i = 0; i < melPoints.Length; i++)
        {
            melPoints[i] = melMin + melStep * i;
        }

        var hzPoints = new float[melPoints.Length];
        for (var i = 0; i < hzPoints.Length; i++)
        {
            hzPoints[i] = MelToHzSlaney(melPoints[i]);
        }

        var binHz = new float[bins];
        var hzScale = (float)sampleRate / nFft;
        for (var bin = 0; bin < bins; bin++)
        {
            binHz[bin] = bin * hzScale;
        }

        for (var mel = 0; mel < nMels; mel++)
        {
            var lower = hzPoints[mel];
            var center = hzPoints[mel + 1];
            var upper = hzPoints[mel + 2];

            var invRise = center > lower ? 1f / (center - lower) : 0f;
            var invFall = upper > center ? 1f / (upper - center) : 0f;
            var norm = upper > lower ? 2f / (upper - lower) : 1f;

            for (var bin = 0; bin < bins; bin++)
            {
                var hz = binHz[bin];
                var weight = 0f;
                if (hz >= lower && hz <= center)
                {
                    weight = (hz - lower) * invRise;
                }
                else if (hz > center && hz <= upper)
                {
                    weight = (upper - hz) * invFall;
                }

                filters[mel, bin] = MathF.Max(0f, weight * norm);
            }
        }

        return filters;
    }

    private static float HzToMelSlaney(float hz)
    {
        const float fSp = 200f / 3f;
        const float minLogHz = 1000f;
        const float minLogMel = minLogHz / fSp; // 15
        const float logStep = 0.06875178f; // ln(6.4) / 27

        if (hz < minLogHz)
        {
            return hz / fSp;
        }

        return minLogMel + MathF.Log(hz / minLogHz) / logStep;
    }

    private static float MelToHzSlaney(float mel)
    {
        const float fSp = 200f / 3f;
        const float minLogHz = 1000f;
        const float minLogMel = minLogHz / fSp; // 15
        const float logStep = 0.06875178f; // ln(6.4) / 27

        if (mel < minLogMel)
        {
            return mel * fSp;
        }

        return minLogHz * MathF.Exp(logStep * (mel - minLogMel));
    }
}
