namespace HifiSampler.Core.Stft;

internal static class FftDispatcher
{
    public static void Transform(float[] real, float[] imag, bool inverse)
    {
        var n = real.Length;
        if (imag.Length != n)
        {
            throw new ArgumentException("Real and imaginary buffers must have the same length.");
        }

        if (n < 1)
        {
            throw new ArgumentException("FFT size must be positive.");
        }

        if (Radix2Fft.IsPowerOfTwo(n))
        {
            Radix2Fft.Transform(real, imag, inverse);
            return;
        }

        BluesteinFft.Transform(real, imag, inverse);
    }
}
