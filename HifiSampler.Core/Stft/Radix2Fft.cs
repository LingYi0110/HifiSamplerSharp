using System.Collections.Concurrent;
using System.Numerics;

namespace HifiSampler.Core.Stft;

internal static class Radix2Fft
{
    private sealed class Radix2Plan(int levels, int[] bitReversed, float[][] stageTwiddleReal, float[][] stageTwiddleImag)
    {
        public int Levels { get; } = levels;
        public int[] BitReversed { get; } = bitReversed;
        public float[][] StageTwiddleReal { get; } = stageTwiddleReal;
        public float[][] StageTwiddleImag { get; } = stageTwiddleImag;
    }

    private static readonly ConcurrentDictionary<int, Radix2Plan> PlanCache = new();

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

        var plan = PlanCache.GetOrAdd(n, CreatePlan);
        var bitReversed = plan.BitReversed;

        for (var i = 0; i < n; i++)
        {
            var j = bitReversed[i];
            if (j <= i)
            {
                continue;
            }

            var tempReal = real[i];
            real[i] = real[j];
            real[j] = tempReal;

            var tempImag = imag[i];
            imag[i] = imag[j];
            imag[j] = tempImag;
        }

        var simd = Vector<float>.Count;
        for (int stage = 0, size = 2; stage < plan.Levels; stage++, size <<= 1)
        {
            var halfSize = size >> 1;
            var stageTwiddleReal = plan.StageTwiddleReal[stage];
            var stageTwiddleImag = plan.StageTwiddleImag[stage];

            for (var baseIndex = 0; baseIndex < n; baseIndex += size)
            {
                if (inverse)
                {
                    var j = 0;
                    for (; j <= halfSize - simd; j += simd)
                    {
                        var left = baseIndex + j;
                        var right = left + halfSize;
                        var leftReal = new Vector<float>(real, left);
                        var leftImag = new Vector<float>(imag, left);
                        var rightReal = new Vector<float>(real, right);
                        var rightImag = new Vector<float>(imag, right);
                        var wReal = new Vector<float>(stageTwiddleReal, j);
                        var wImag = -new Vector<float>(stageTwiddleImag, j);
                        var tReal = wReal * rightReal - wImag * rightImag;
                        var tImag = wReal * rightImag + wImag * rightReal;
                        (leftReal - tReal).CopyTo(real, right);
                        (leftImag - tImag).CopyTo(imag, right);
                        (leftReal + tReal).CopyTo(real, left);
                        (leftImag + tImag).CopyTo(imag, left);
                    }

                    for (; j < halfSize; j++)
                    {
                        var left = baseIndex + j;
                        var right = left + halfSize;
                        var wReal = stageTwiddleReal[j];
                        var wImag = -stageTwiddleImag[j];
                        var tReal = wReal * real[right] - wImag * imag[right];
                        var tImag = wReal * imag[right] + wImag * real[right];
                        real[right] = real[left] - tReal;
                        imag[right] = imag[left] - tImag;
                        real[left] += tReal;
                        imag[left] += tImag;
                    }
                }
                else
                {
                    var j = 0;
                    for (; j <= halfSize - simd; j += simd)
                    {
                        var left = baseIndex + j;
                        var right = left + halfSize;
                        var leftReal = new Vector<float>(real, left);
                        var leftImag = new Vector<float>(imag, left);
                        var rightReal = new Vector<float>(real, right);
                        var rightImag = new Vector<float>(imag, right);
                        var wReal = new Vector<float>(stageTwiddleReal, j);
                        var wImag = new Vector<float>(stageTwiddleImag, j);
                        var tReal = wReal * rightReal - wImag * rightImag;
                        var tImag = wReal * rightImag + wImag * rightReal;
                        (leftReal - tReal).CopyTo(real, right);
                        (leftImag - tImag).CopyTo(imag, right);
                        (leftReal + tReal).CopyTo(real, left);
                        (leftImag + tImag).CopyTo(imag, left);
                    }

                    for (; j < halfSize; j++)
                    {
                        var left = baseIndex + j;
                        var right = left + halfSize;
                        var wReal = stageTwiddleReal[j];
                        var wImag = stageTwiddleImag[j];
                        var tReal = wReal * real[right] - wImag * imag[right];
                        var tImag = wReal * imag[right] + wImag * real[right];
                        real[right] = real[left] - tReal;
                        imag[right] = imag[left] - tImag;
                        real[left] += tReal;
                        imag[left] += tImag;
                    }
                }
            }
        }

        if (!inverse)
        {
            return;
        }

        var invN = 1f / n;
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

    private static Radix2Plan CreatePlan(int n)
    {
        var levels = 0;
        for (var temp = n; temp > 1; temp >>= 1)
        {
            levels++;
        }

        var bitReversed = new int[n];
        for (var i = 0; i < n; i++)
        {
            bitReversed[i] = (int)(ReverseBits((uint)i) >> (32 - levels));
        }

        var stageTwiddleReal = new float[levels][];
        var stageTwiddleImag = new float[levels][];
        for (int stage = 0, size = 2; stage < levels; stage++, size <<= 1)
        {
            var angle = -2f * MathF.PI / size;
            var wMulReal = MathF.Cos(angle);
            var wMulImag = MathF.Sin(angle);
            var halfSize = size >> 1;
            var twiddleReal = new float[halfSize];
            var twiddleImag = new float[halfSize];
            twiddleReal[0] = 1f;
            twiddleImag[0] = 0f;
            for (var j = 1; j < halfSize; j++)
            {
                var previousReal = twiddleReal[j - 1];
                var previousImag = twiddleImag[j - 1];
                twiddleReal[j] = previousReal * wMulReal - previousImag * wMulImag;
                twiddleImag[j] = previousReal * wMulImag + previousImag * wMulReal;
            }

            stageTwiddleReal[stage] = twiddleReal;
            stageTwiddleImag[stage] = twiddleImag;
        }

        return new Radix2Plan(levels, bitReversed, stageTwiddleReal, stageTwiddleImag);
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
