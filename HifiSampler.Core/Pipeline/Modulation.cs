namespace HifiSampler.Core.Pipeline;

public static class Modulation
{
    public static float[] ApplyAmplitudeFromPitch(
        float[] render,
        double[] tMel,
        float[] f0Mel,
        double newStart,
        double newEnd,
        int aFlag)
    {
        if (render.Length == 0 || tMel.Length < 2 || f0Mel.Length < 2 || aFlag == 0)
        {
            return render;
        }

        var gainAtMelFrames = new double[tMel.Length];
        var midi = f0Mel.Select(ToMidi).ToArray();
        for (var i = 0; i < tMel.Length; i++)
        {
            var prev = midi[Math.Max(0, i - 1)];
            var next = midi[Math.Min(midi.Length - 1, i + 1)];
            var dt = Math.Max(1e-8, tMel[Math.Min(tMel.Length - 1, i + 1)] - tMel[Math.Max(0, i - 1)]);
            var derivative = (next - prev) / dt;
            gainAtMelFrames[i] = Math.Pow(5, 1e-4 * Math.Clamp(aFlag, -100, 100) * derivative);
        }

        var n = render.Length;
        for (var i = 0; i < n; i++)
        {
            var t = newStart + (newEnd - newStart) * i / n;
            var gain = LinearSample(tMel, gainAtMelFrames, t);
            render[i] = (float)(render[i] * gain);
        }

        return render;
    }

    public static float[] ApplyGrowl(float[] audio, int sampleRate, int hgFlag)
    {
        if (audio.Length == 0 || hgFlag <= 0)
        {
            return audio;
        }

        var strength = Math.Clamp(hgFlag, 0, 100) / 100f;
        var hp = Highpass(audio, sampleRate, 400f);
        var result = new float[audio.Length];
        var freq = 80.0;
        for (var i = 0; i < audio.Length; i++)
        {
            var t = i / (double)sampleRate;
            var lfo = Math.Sin(2 * Math.PI * freq * t) >= 0 ? 1f : -1f;
            var band = hp[i] * (1f + 0.15f * strength * lfo);
            var low = audio[i] - hp[i];
            result[i] = low + band;
        }

        return result;
    }

    private static float[] Highpass(float[] audio, int sampleRate, float cutoffHz)
    {
        var output = new float[audio.Length];
        if (audio.Length == 0)
        {
            return output;
        }

        var rc = 1.0 / (2 * Math.PI * cutoffHz);
        var dt = 1.0 / Math.Max(1, sampleRate);
        var alpha = rc / (rc + dt);
        output[0] = audio[0];
        for (var i = 1; i < audio.Length; i++)
        {
            output[i] = (float)(alpha * (output[i - 1] + audio[i] - audio[i - 1]));
        }

        return output;
    }

    private static double ToMidi(float hz)
    {
        if (hz <= 1e-8f)
        {
            return 0;
        }

        return 69.0 + 12.0 * Math.Log2(hz / 440.0);
    }

    private static double LinearSample(double[] x, double[] y, double t)
    {
        if (t <= x[0])
        {
            return y[0];
        }

        if (t >= x[^1])
        {
            return y[^1];
        }

        var idx = Array.BinarySearch(x, t);
        if (idx >= 0)
        {
            return y[idx];
        }

        idx = ~idx;
        var x0 = x[idx - 1];
        var x1 = x[idx];
        var y0 = y[idx - 1];
        var y1 = y[idx];
        var ratio = (t - x0) / Math.Max(1e-8, x1 - x0);
        return y0 + (y1 - y0) * ratio;
    }
}
