namespace HifiSampler.Core.Audio;

public static class ResamplingUtils
{
    public static float[] Clip(float[] samples, float min, float max)
    {
        return samples.Select(x => Math.Clamp(x, min, max)).ToArray();
    }

    public static double[] LinearInterpolate(double[] x, double[] y, double[] target)
    {
        if (x.Length == 0 || y.Length == 0 || x.Length != y.Length)
        {
            return [];
        }

        var result = new double[target.Length];
        var cursor = 0;
        for (var i = 0; i < target.Length; i++)
        {
            var t = target[i];
            while (cursor + 1 < x.Length && x[cursor + 1] < t)
            {
                cursor++;
            }

            if (cursor + 1 >= x.Length)
            {
                result[i] = y[^1];
                continue;
            }

            var t0 = x[cursor];
            var t1 = x[cursor + 1];
            var y0 = y[cursor];
            var y1 = y[cursor + 1];
            var ratio = (t - t0) / Math.Max(t1 - t0, 1e-8);
            result[i] = y0 + (y1 - y0) * ratio;
        }

        return result;
    }
}
