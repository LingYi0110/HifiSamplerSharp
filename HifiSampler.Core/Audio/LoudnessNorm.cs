namespace HifiSampler.Core.Audio;

public static class LoudnessNorm
{
    public static float[] Normalize(float[] waveform, int sampleRate)
    {
        _ = sampleRate;
        if (waveform.Length == 0)
        {
            return waveform;
        }

        var peak = waveform.Max(MathF.Abs);
        if (peak < 1e-6f)
        {
            return waveform;
        }

        var gain = 0.95f / peak;
        return waveform.Select(x => x * gain).ToArray();
    }
}
