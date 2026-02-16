using System.Collections.Concurrent;
using System.Numerics;

namespace HifiSampler.Core.Stft;

internal static class BluesteinFft
{
    private sealed class WorkBuffer(int convolutionLength)
    {
        public float[] Real { get; } = new float[convolutionLength];
        public float[] Imag { get; } = new float[convolutionLength];
    }

    private sealed class BluesteinPlan(
        int length,
        int convolutionLength,
        float[] chirpCos,
        float[] chirpSin,
        float[] forwardKernelReal,
        float[] forwardKernelImag,
        float[] inverseKernelReal,
        float[] inverseKernelImag,
        ConcurrentBag<WorkBuffer> workspaces)
    {
        public int Length { get; } = length;
        public int ConvolutionLength { get; } = convolutionLength;
        public float[] ChirpCos { get; } = chirpCos;
        public float[] ChirpSin { get; } = chirpSin;
        public float[] ForwardKernelReal { get; } = forwardKernelReal;
        public float[] ForwardKernelImag { get; } = forwardKernelImag;
        public float[] InverseKernelReal { get; } = inverseKernelReal;
        public float[] InverseKernelImag { get; } = inverseKernelImag;
        public ConcurrentBag<WorkBuffer> Workspaces { get; } = workspaces;
    }

    private static readonly ConcurrentDictionary<int, BluesteinPlan> PlanCache = new();

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

        var plan = PlanCache.GetOrAdd(n, CreatePlan);
        var m = plan.ConvolutionLength;
        if (!plan.Workspaces.TryTake(out var workspace))
        {
            workspace = new WorkBuffer(m);
        }

        var workReal = workspace.Real;
        var workImag = workspace.Imag;
        var chirpCos = plan.ChirpCos;
        var chirpSin = plan.ChirpSin;

        try
        {
            if (inverse)
            {
                // Multiply by exp(+i * pi * i^2 / n).
                ApplyInputChirpInverse(real, imag, chirpCos, chirpSin, workReal, workImag, n);
            }
            else
            {
                // Multiply by exp(-i * pi * i^2 / n).
                ApplyInputChirpForward(real, imag, chirpCos, chirpSin, workReal, workImag, n);
            }

            var tailLength = m - n;
            if (tailLength > 0)
            {
                Array.Clear(workReal, n, tailLength);
                Array.Clear(workImag, n, tailLength);
            }

            Radix2Fft.Transform(workReal, workImag, inverse: false);

            var kernelReal = inverse ? plan.InverseKernelReal : plan.ForwardKernelReal;
            var kernelImag = inverse ? plan.InverseKernelImag : plan.ForwardKernelImag;
            MultiplySpectrumByKernel(workReal, workImag, kernelReal, kernelImag, m);

            Radix2Fft.Transform(workReal, workImag, inverse: true);

            if (inverse)
            {
                // Multiply by exp(+i * pi * i^2 / n), and apply 1/n scaling.
                ApplyOutputChirpInverse(workReal, workImag, chirpCos, chirpSin, real, imag, n, 1f / n);
            }
            else
            {
                // Multiply by exp(-i * pi * i^2 / n).
                ApplyOutputChirpForward(workReal, workImag, chirpCos, chirpSin, real, imag, n);
            }
        }
        finally
        {
            plan.Workspaces.Add(workspace);
        }
    }

    private static BluesteinPlan CreatePlan(int n)
    {
        var m = NextPowerOfTwo((2 * n) - 1);
        var chirpCos = new float[n];
        var chirpSin = new float[n];

        var forwardKernelReal = new float[m];
        var forwardKernelImag = new float[m];
        var inverseKernelReal = new float[m];
        var inverseKernelImag = new float[m];

        for (var i = 0; i < n; i++)
        {
            var phase = ComputePhase(i, n);
            var cos = (float)Math.Cos(phase);
            var sin = (float)Math.Sin(phase);
            chirpCos[i] = cos;
            chirpSin[i] = sin;

            // forward b[i] = exp(+i * phase)
            forwardKernelReal[i] = cos;
            forwardKernelImag[i] = sin;
            // inverse b[i] = exp(-i * phase)
            inverseKernelReal[i] = cos;
            inverseKernelImag[i] = -sin;

            if (i == 0)
            {
                continue;
            }

            var mirrored = m - i;
            forwardKernelReal[mirrored] = cos;
            forwardKernelImag[mirrored] = sin;
            inverseKernelReal[mirrored] = cos;
            inverseKernelImag[mirrored] = -sin;
        }

        Radix2Fft.Transform(forwardKernelReal, forwardKernelImag, inverse: false);
        Radix2Fft.Transform(inverseKernelReal, inverseKernelImag, inverse: false);

        return new BluesteinPlan(
            n,
            m,
            chirpCos,
            chirpSin,
            forwardKernelReal,
            forwardKernelImag,
            inverseKernelReal,
            inverseKernelImag,
            new ConcurrentBag<WorkBuffer>());
    }

    private static void ApplyInputChirpForward(
        float[] inputReal,
        float[] inputImag,
        float[] chirpCos,
        float[] chirpSin,
        float[] workReal,
        float[] workImag,
        int count)
    {
        var simd = Vector<float>.Count;
        var i = 0;
        for (; i <= count - simd; i += simd)
        {
            var r = new Vector<float>(inputReal, i);
            var im = new Vector<float>(inputImag, i);
            var cos = new Vector<float>(chirpCos, i);
            var sin = new Vector<float>(chirpSin, i);
            (r * cos + im * sin).CopyTo(workReal, i);
            (im * cos - r * sin).CopyTo(workImag, i);
        }

        for (; i < count; i++)
        {
            var r = inputReal[i];
            var im = inputImag[i];
            var cos = chirpCos[i];
            var sin = chirpSin[i];
            workReal[i] = r * cos + im * sin;
            workImag[i] = im * cos - r * sin;
        }
    }

    private static void ApplyInputChirpInverse(
        float[] inputReal,
        float[] inputImag,
        float[] chirpCos,
        float[] chirpSin,
        float[] workReal,
        float[] workImag,
        int count)
    {
        var simd = Vector<float>.Count;
        var i = 0;
        for (; i <= count - simd; i += simd)
        {
            var r = new Vector<float>(inputReal, i);
            var im = new Vector<float>(inputImag, i);
            var cos = new Vector<float>(chirpCos, i);
            var sin = new Vector<float>(chirpSin, i);
            (r * cos - im * sin).CopyTo(workReal, i);
            (r * sin + im * cos).CopyTo(workImag, i);
        }

        for (; i < count; i++)
        {
            var r = inputReal[i];
            var im = inputImag[i];
            var cos = chirpCos[i];
            var sin = chirpSin[i];
            workReal[i] = r * cos - im * sin;
            workImag[i] = r * sin + im * cos;
        }
    }

    private static void MultiplySpectrumByKernel(
        float[] workReal,
        float[] workImag,
        float[] kernelReal,
        float[] kernelImag,
        int count)
    {
        var simd = Vector<float>.Count;
        var i = 0;
        for (; i <= count - simd; i += simd)
        {
            var rr = new Vector<float>(workReal, i);
            var ii = new Vector<float>(workImag, i);
            var kr = new Vector<float>(kernelReal, i);
            var ki = new Vector<float>(kernelImag, i);

            var outReal = rr * kr - ii * ki;
            var outImag = rr * ki + ii * kr;
            outReal.CopyTo(workReal, i);
            outImag.CopyTo(workImag, i);
        }

        for (; i < count; i++)
        {
            var rr = workReal[i];
            var ii = workImag[i];
            var kr = kernelReal[i];
            var ki = kernelImag[i];
            workReal[i] = rr * kr - ii * ki;
            workImag[i] = rr * ki + ii * kr;
        }
    }

    private static void ApplyOutputChirpForward(
        float[] workReal,
        float[] workImag,
        float[] chirpCos,
        float[] chirpSin,
        float[] outputReal,
        float[] outputImag,
        int count)
    {
        var simd = Vector<float>.Count;
        var i = 0;
        for (; i <= count - simd; i += simd)
        {
            var rr = new Vector<float>(workReal, i);
            var ii = new Vector<float>(workImag, i);
            var cos = new Vector<float>(chirpCos, i);
            var sin = new Vector<float>(chirpSin, i);
            (rr * cos + ii * sin).CopyTo(outputReal, i);
            (ii * cos - rr * sin).CopyTo(outputImag, i);
        }

        for (; i < count; i++)
        {
            var rr = workReal[i];
            var ii = workImag[i];
            var cos = chirpCos[i];
            var sin = chirpSin[i];
            outputReal[i] = rr * cos + ii * sin;
            outputImag[i] = ii * cos - rr * sin;
        }
    }

    private static void ApplyOutputChirpInverse(
        float[] workReal,
        float[] workImag,
        float[] chirpCos,
        float[] chirpSin,
        float[] outputReal,
        float[] outputImag,
        int count,
        float scale)
    {
        var simd = Vector<float>.Count;
        var scaleVec = new Vector<float>(scale);
        var i = 0;
        for (; i <= count - simd; i += simd)
        {
            var rr = new Vector<float>(workReal, i);
            var ii = new Vector<float>(workImag, i);
            var cos = new Vector<float>(chirpCos, i);
            var sin = new Vector<float>(chirpSin, i);
            ((rr * cos - ii * sin) * scaleVec).CopyTo(outputReal, i);
            ((rr * sin + ii * cos) * scaleVec).CopyTo(outputImag, i);
        }

        for (; i < count; i++)
        {
            var rr = workReal[i];
            var ii = workImag[i];
            var cos = chirpCos[i];
            var sin = chirpSin[i];
            outputReal[i] = (rr * cos - ii * sin) * scale;
            outputImag[i] = (rr * sin + ii * cos) * scale;
        }
    }

    private static int NextPowerOfTwo(int value)
    {
        if (value <= 1)
        {
            return 1;
        }

        var p = 1;
        while (p < value && p <= (1 << 29))
        {
            p <<= 1;
        }

        if (p < value)
        {
            throw new ArgumentOutOfRangeException(nameof(value), "FFT size is too large.");
        }

        return p;
    }

    private static double ComputePhase(int index, int n)
    {
        // exp(i * pi * k^2 / n) has period 2n. Use modular arithmetic to avoid overflow and precision loss.
        var period = 2L * n;
        var reduced = index % period;
        var squareMod = (reduced * reduced) % period;
        return Math.PI * squareMod / n;
    }
}
