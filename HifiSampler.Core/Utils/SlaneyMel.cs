using System.Numerics;

namespace HifiSampler.Core.Utils;

internal static class SlaneyMel
{
    public static float[] BuildMelFilterBank(
        int nMels,
        int nFft,
        int sampleRate,
        float fMin,
        float fMax)
    {
        var bins = nFft / 2 + 1;
        var filters = new float[nMels * bins];

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

            var rowOffset = mel * bins;
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

                filters[rowOffset + bin] = MathF.Max(0f, weight * norm);
            }
        }

        return filters;
    }

    public static float[] Magnitude(float[] real, float[] imaginary)
    {
        var length = Math.Min(real.Length, imaginary.Length);
        var output = new float[length];

        var simd = Vector<float>.Count;
        var i = 0;
        for (; i <= length - simd; i += simd)
        {
            var r = new Vector<float>(real, i);
            var im = new Vector<float>(imaginary, i);
            Vector.SquareRoot(r * r + im * im).CopyTo(output, i);
        }

        for (; i < length; i++)
        {
            output[i] = MathF.Sqrt(real[i] * real[i] + imaginary[i] * imaginary[i]);
        }

        return output;
    }

    public static void ComplexMultiply(
        float[] inputReal,
        float[] inputImag,
        float[] maskReal,
        float[] maskImag,
        float[] outputReal,
        float[] outputImag)
    {
        var length = inputReal.Length;
        if (inputImag.Length != length ||
            maskReal.Length != length ||
            maskImag.Length != length ||
            outputReal.Length != length ||
            outputImag.Length != length)
        {
            throw new ArgumentException("All complex buffers must have the same length.");
        }

        var simd = Vector<float>.Count;
        var i = 0;
        for (; i <= length - simd; i += simd)
        {
            var ar = new Vector<float>(inputReal, i);
            var ai = new Vector<float>(inputImag, i);
            var br = new Vector<float>(maskReal, i);
            var bi = new Vector<float>(maskImag, i);
            (ar * br - ai * bi).CopyTo(outputReal, i);
            (ar * bi + ai * br).CopyTo(outputImag, i);
        }

        for (; i < length; i++)
        {
            var real = inputReal[i];
            var imag = inputImag[i];
            var maskR = maskReal[i];
            var maskI = maskImag[i];
            outputReal[i] = real * maskR - imag * maskI;
            outputImag[i] = real * maskI + imag * maskR;
        }
    }

    public static float[,] Multiply(
        float[] left,
        int leftRows,
        int sharedDim,
        float[] right,
        int rightCols)
    {
        if (left.Length != leftRows * sharedDim)
        {
            throw new ArgumentException("Left matrix dimensions do not match the buffer length.");
        }

        if (right.Length != sharedDim * rightCols)
        {
            throw new ArgumentException("Right matrix dimensions do not match the buffer length.");
        }

        var outputFlat = new float[leftRows * rightCols];
        Parallel.For(0, leftRows, row =>
        {
            var outputRow = outputFlat.AsSpan(row * rightCols, rightCols);
            outputRow.Clear();

            var leftOffset = row * sharedDim;
            for (var k = 0; k < sharedDim; k++)
            {
                var coefficient = left[leftOffset + k];
                if (MathF.Abs(coefficient) <= float.Epsilon)
                {
                    continue;
                }

                var rightRow = right.AsSpan(k * rightCols, rightCols);
                Axpy(coefficient, rightRow, outputRow);
            }
        });

        var output = new float[leftRows, rightCols];
        Buffer.BlockCopy(outputFlat, 0, output, 0, outputFlat.Length * sizeof(float));
        return output;
    }

    public static void ScaleInPlace(float[] values, float scale)
    {
        var simd = Vector<float>.Count;
        var vectorScale = new Vector<float>(scale);
        var i = 0;
        for (; i <= values.Length - simd; i += simd)
        {
            (new Vector<float>(values, i) * vectorScale).CopyTo(values, i);
        }

        for (; i < values.Length; i++)
        {
            values[i] *= scale;
        }
    }

    private static void Axpy(float coefficient, ReadOnlySpan<float> x, Span<float> y)
    {
        var simd = Vector<float>.Count;
        var vectorScale = new Vector<float>(coefficient);
        var i = 0;
        for (; i <= x.Length - simd; i += simd)
        {
            var xv = new Vector<float>(x.Slice(i, simd));
            var yv = new Vector<float>(y.Slice(i, simd));
            (yv + xv * vectorScale).CopyTo(y.Slice(i, simd));
        }

        for (; i < x.Length; i++)
        {
            y[i] += coefficient * x[i];
        }
    }

    private static float HzToMelSlaney(float hz)
    {
        // librosa/auditory-toolbox style Slaney mel scale
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
