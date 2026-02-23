using System.Collections.Concurrent;
using System.Numerics;

namespace HifiSampler.Core.Stft;

internal static class Radix2Fft
{
    private sealed class Radix2Plan(
        int levels,
        int[] bitReversed,
        int[] stageOffsets,
        float[] twiddleReal,
        float[] twiddleImagForward,
        float[] twiddleImagInverse)
    {
        public int Levels { get; } = levels;
        public int[] BitReversed { get; } = bitReversed;
        public int[] StageOffsets { get; } = stageOffsets;
        public float[] TwiddleReal { get; } = twiddleReal;
        public float[] TwiddleImagForward { get; } = twiddleImagForward;
        public float[] TwiddleImagInverse { get; } = twiddleImagInverse;
    }

    private static readonly ConcurrentDictionary<int, Radix2Plan> PlanCache = new();

    public static bool IsPowerOfTwo(int value) => value > 0 && (value & (value - 1)) == 0;

    public static void Transform(float[] real, float[] imag, bool inverse)
    {
        ArgumentNullException.ThrowIfNull(real);
        ArgumentNullException.ThrowIfNull(imag);
        Transform(real.AsSpan(), imag.AsSpan(), inverse);
    }

    public static void Transform(Span<float> real, Span<float> imag, bool inverse)
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
        BitReversePermute(real, imag, plan.BitReversed);

        if (inverse)
        {
            TransformCore(real, imag, n, plan, plan.TwiddleImagInverse);
            ScaleInverse(real, imag, n);
            return;
        }

        TransformCore(real, imag, n, plan, plan.TwiddleImagForward);
    }

    private static void BitReversePermute(Span<float> real, Span<float> imag, int[] bitReversed)
    {
        for (var i = 0; i < bitReversed.Length; i++)
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
    }

    private static void TransformCore(
        Span<float> real,
        Span<float> imag,
        int n,
        Radix2Plan plan,
        float[] twiddleImag)
    {
        var simd = Vector<float>.Count;
        var twiddleReal = plan.TwiddleReal;
        var stageOffsets = plan.StageOffsets;

        for (int stage = 0, size = 2; stage < plan.Levels; stage++, size <<= 1)
        {
            var halfSize = size >> 1;
            var stageOffset = stageOffsets[stage];

            for (var baseIndex = 0; baseIndex < n; baseIndex += size)
            {
                var j = 0;
                for (; j <= halfSize - simd; j += simd)
                {
                    var left = baseIndex + j;
                    var right = left + halfSize;

                    var leftReal = new Vector<float>(real.Slice(left));
                    var leftImag = new Vector<float>(imag.Slice(left));
                    var rightReal = new Vector<float>(real.Slice(right));
                    var rightImag = new Vector<float>(imag.Slice(right));
                    var wReal = new Vector<float>(twiddleReal, stageOffset + j);
                    var wImag = new Vector<float>(twiddleImag, stageOffset + j);
                    var tReal = wReal * rightReal - wImag * rightImag;
                    var tImag = wReal * rightImag + wImag * rightReal;

                    (leftReal - tReal).CopyTo(real.Slice(right));
                    (leftImag - tImag).CopyTo(imag.Slice(right));
                    (leftReal + tReal).CopyTo(real.Slice(left));
                    (leftImag + tImag).CopyTo(imag.Slice(left));
                }

                for (; j < halfSize; j++)
                {
                    var left = baseIndex + j;
                    var right = left + halfSize;
                    var wReal = twiddleReal[stageOffset + j];
                    var wImag = twiddleImag[stageOffset + j];
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

    private static void ScaleInverse(Span<float> real, Span<float> imag, int n)
    {
        var invN = 1f / n;
        var simd = Vector<float>.Count;
        var scale = new Vector<float>(invN);
        var idx = 0;

        for (; idx <= n - simd; idx += simd)
        {
            (new Vector<float>(real.Slice(idx)) * scale).CopyTo(real.Slice(idx));
            (new Vector<float>(imag.Slice(idx)) * scale).CopyTo(imag.Slice(idx));
        }

        for (; idx < n; idx++)
        {
            real[idx] *= invN;
            imag[idx] *= invN;
        }
    }

    private static Radix2Plan CreatePlan(int n)
    {
        var levels = BitOperations.Log2((uint)n);

        var bitReversed = new int[n];
        for (var i = 0; i < n; i++)
        {
            bitReversed[i] = (int)(ReverseBits((uint)i) >> (32 - levels));
        }

        var stageOffsets = new int[levels];
        var twiddleCount = n - 1;
        var twiddleReal = new float[twiddleCount];
        var twiddleImagForward = new float[twiddleCount];
        var twiddleImagInverse = new float[twiddleCount];

        var twiddleOffset = 0;
        for (int stage = 0, size = 2; stage < levels; stage++, size <<= 1)
        {
            stageOffsets[stage] = twiddleOffset;
            var halfSize = size >> 1;
            var angle = -2f * MathF.PI / size;
            var wMulReal = MathF.Cos(angle);
            var wMulImag = MathF.Sin(angle);

            twiddleReal[twiddleOffset] = 1f;
            twiddleImagForward[twiddleOffset] = 0f;
            twiddleImagInverse[twiddleOffset] = 0f;
            for (var j = 1; j < halfSize; j++)
            {
                var previousReal = twiddleReal[twiddleOffset + j - 1];
                var previousImag = twiddleImagForward[twiddleOffset + j - 1];
                var nextReal = previousReal * wMulReal - previousImag * wMulImag;
                var nextImag = previousReal * wMulImag + previousImag * wMulReal;
                twiddleReal[twiddleOffset + j] = nextReal;
                twiddleImagForward[twiddleOffset + j] = nextImag;
                twiddleImagInverse[twiddleOffset + j] = -nextImag;
            }

            twiddleOffset += halfSize;
        }

        return new Radix2Plan(
            levels,
            bitReversed,
            stageOffsets,
            twiddleReal,
            twiddleImagForward,
            twiddleImagInverse);
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
