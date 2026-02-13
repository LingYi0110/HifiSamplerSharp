using System.Collections.Concurrent;

namespace HifiSampler.Core.Stft;

internal static class BluesteinFft
{
    private sealed class BluesteinPlan(
        int length,
        int convolutionLength,
        float[] chirpCos,
        float[] chirpSin,
        float[] forwardKernelReal,
        float[] forwardKernelImag,
        float[] inverseKernelReal,
        float[] inverseKernelImag)
    {
        public int Length { get; } = length;
        public int ConvolutionLength { get; } = convolutionLength;
        public float[] ChirpCos { get; } = chirpCos;
        public float[] ChirpSin { get; } = chirpSin;
        public float[] ForwardKernelReal { get; } = forwardKernelReal;
        public float[] ForwardKernelImag { get; } = forwardKernelImag;
        public float[] InverseKernelReal { get; } = inverseKernelReal;
        public float[] InverseKernelImag { get; } = inverseKernelImag;
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

        var workReal = new float[m];
        var workImag = new float[m];
        var chirpCos = plan.ChirpCos;
        var chirpSin = plan.ChirpSin;

        for (var i = 0; i < n; i++)
        {
            var r = real[i];
            var im = imag[i];
            var cos = chirpCos[i];
            var sin = chirpSin[i];

            if (inverse)
            {
                // Multiply by exp(+i * pi * i^2 / n).
                workReal[i] = r * cos - im * sin;
                workImag[i] = r * sin + im * cos;
            }
            else
            {
                // Multiply by exp(-i * pi * i^2 / n).
                workReal[i] = r * cos + im * sin;
                workImag[i] = im * cos - r * sin;
            }
        }

        Radix2Fft.Transform(workReal, workImag, inverse: false);

        var kernelReal = inverse ? plan.InverseKernelReal : plan.ForwardKernelReal;
        var kernelImag = inverse ? plan.InverseKernelImag : plan.ForwardKernelImag;
        for (var i = 0; i < m; i++)
        {
            var rr = workReal[i];
            var ii = workImag[i];
            var kr = kernelReal[i];
            var ki = kernelImag[i];
            workReal[i] = rr * kr - ii * ki;
            workImag[i] = rr * ki + ii * kr;
        }

        Radix2Fft.Transform(workReal, workImag, inverse: true);

        var invN = 1f / n;
        for (var i = 0; i < n; i++)
        {
            var rr = workReal[i];
            var ii = workImag[i];
            var cos = chirpCos[i];
            var sin = chirpSin[i];

            if (inverse)
            {
                // Multiply by exp(+i * pi * i^2 / n).
                var outReal = rr * cos - ii * sin;
                var outImag = rr * sin + ii * cos;
                real[i] = outReal * invN;
                imag[i] = outImag * invN;
            }
            else
            {
                // Multiply by exp(-i * pi * i^2 / n).
                real[i] = rr * cos + ii * sin;
                imag[i] = ii * cos - rr * sin;
            }
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
            inverseKernelImag);
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
