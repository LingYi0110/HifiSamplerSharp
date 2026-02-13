namespace HifiSampler.Core.Audio;

public static class LoudnessNorm
{
    public static float[] Normalize(float[] waveform, int sampleRate, int strength = 100)
    {
        _ = sampleRate;
        strength = Math.Clamp(strength, 0, 100);

        if (waveform.Length == 0)
        {
            return waveform;
        }

        if (strength == 0)
        {
            return waveform;
        }

        double sumSq = 0;
        for (var i = 0; i < waveform.Length; i++)
        {
            sumSq += waveform[i] * waveform[i];
        }

        var rms = (float)Math.Sqrt(sumSq / Math.Max(1, waveform.Length));
        if (rms < 1e-8f)
        {
            return waveform;
        }

        // Approximate note loudness target (-16 dBFS-like RMS reference).
        const float targetRms = 0.15848932f; // 10^(-16/20)
        var desiredGain = targetRms / rms;
        var blend = strength / 100f;
        var gain = 1f + (desiredGain - 1f) * blend;

        var output = new float[waveform.Length];
        var peak = 0f;
        for (var i = 0; i < waveform.Length; i++)
        {
            var sample = waveform[i] * gain;
            output[i] = sample;
            var abs = MathF.Abs(sample);
            if (abs > peak)
            {
                peak = abs;
            }
        }

        // Keep a headroom close to -1 dBFS.
        const float peakTarget = 0.8912509f;
        if (peak > peakTarget && peak > 1e-8f)
        {
            var limiter = peakTarget / peak;
            for (var i = 0; i < output.Length; i++)
            {
                output[i] *= limiter;
            }
        }

        return output;
    }
}
