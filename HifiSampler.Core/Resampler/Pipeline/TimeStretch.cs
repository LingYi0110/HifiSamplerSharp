namespace HifiSampler.Core.Resampler.Pipeline;

public static class TimeStretch
{
    public static double Stretch(double t, double consonantBoundary, double scalingRatio, double velocityScale)
    {
        return t < velocityScale * consonantBoundary
            ? t / velocityScale
            : consonantBoundary + (t - velocityScale * consonantBoundary) / scalingRatio;
    }

    public static float[] ReflectLoop(float[] source, int targetLength)
    {
        if (source.Length == 0 || targetLength <= 0)
        {
            return [];
        }

        if (source.Length >= targetLength)
        {
            return source[..targetLength];
        }

        var result = new float[targetLength];
        for (var i = 0; i < targetLength; i++)
        {
            var period = i / source.Length;
            var index = i % source.Length;
            if (period % 2 == 1)
            {
                index = source.Length - 1 - index;
            }

            result[i] = source[index];
        }

        return result;
    }
}
