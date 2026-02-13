using System.Numerics;

namespace HifiSampler.Core.Stft;

internal static class Radix2Fft
{
    public static bool IsPowerOfTwo(int value) => value > 0 && (value & (value - 1)) == 0;

    public static void Transform(float[] real, float[] imag, bool inverse)
    {
        var n = real.Length;
        if (imag.Length != n)
        {
            throw new ArgumentException("Real and imaginary buffers must have the same length.");
        }

        if (!IsPowerOfTwo(n))
        {
            throw new ArgumentException($"FFT size must be power-of-two. Got {n}.");
        }

        var levels = 0;
        for (var temp = n; temp > 1; temp >>= 1)
        {
            levels++;
        }

        for (var i = 0; i < n; i++)
        {
            var j = (int)(ReverseBits((uint)i) >> (32 - levels));
            if (j <= i)
            {
                continue;
            }

            (real[i], real[j]) = (real[j], real[i]);
            (imag[i], imag[j]) = (imag[j], imag[i]);
        }

        for (var size = 2; size <= n; size <<= 1)
        {
            var halfSize = size >> 1;
            var angle = (inverse ? 2f : -2f) * MathF.PI / size;
            var wMulReal = MathF.Cos(angle);
            var wMulImag = MathF.Sin(angle);

            for (var baseIndex = 0; baseIndex < n; baseIndex += size)
            {
                var wReal = 1f;
                var wImag = 0f;
                for (var j = 0; j < halfSize; j++)
                {
                    var left = baseIndex + j;
                    var right = left + halfSize;

                    var tReal = wReal * real[right] - wImag * imag[right];
                    var tImag = wReal * imag[right] + wImag * real[right];

                    real[right] = real[left] - tReal;
                    imag[right] = imag[left] - tImag;
                    real[left] += tReal;
                    imag[left] += tImag;

                    var oldWReal = wReal;
                    wReal = oldWReal * wMulReal - wImag * wMulImag;
                    wImag = oldWReal * wMulImag + wImag * wMulReal;
                }
            }
        }

        if (!inverse)
        {
            return;
        }

        var invN = 1f / n;
        var simd = Vector<float>.Count;
        var scale = new Vector<float>(invN);
        var idx = 0;
        for (; idx <= n - simd; idx += simd)
        {
            (new Vector<float>(real, idx) * scale).CopyTo(real, idx);
            (new Vector<float>(imag, idx) * scale).CopyTo(imag, idx);
        }

        for (; idx < n; idx++)
        {
            real[idx] *= invN;
            imag[idx] *= invN;
        }
    }

    private static uint ReverseBits(uint value)
    {
        value = ((value & 0x55555555u) << 1) | ((value >> 1) & 0x55555555u);
        value = ((value & 0x33333333u) << 2) | ((value >> 2) & 0x33333333u);
        value = ((value & 0x0F0F0F0Fu) << 4) | ((value >> 4) & 0x0F0F0F0Fu);
        value = ((value & 0x00FF00FFu) << 8) | ((value >> 8) & 0x00FF00FFu);
        value = (value << 16) | (value >> 16);
        return value;
    }
}
