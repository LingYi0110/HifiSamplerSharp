using System.Buffers;
using System.Numerics;

namespace HifiSampler.Core.Stft;

internal readonly record struct ComplexSpectrogram(
    float[] Real,
    float[] Imaginary,
    int Bins,
    int Frames);

internal static class StftEngine
{
    private const int ParallelFrameThreshold = 32;

    private sealed class PooledFrameBuffer : IDisposable
    {
        private readonly int _fftSize;

        public PooledFrameBuffer(int fftSize)
        {
            _fftSize = fftSize;
            Real = ArrayPool<float>.Shared.Rent(fftSize);
            Imaginary = ArrayPool<float>.Shared.Rent(fftSize);
        }

        public float[] Real { get; }
        public float[] Imaginary { get; }

        public Span<float> RealSpan => Real.AsSpan(0, _fftSize);
        public Span<float> ImaginarySpan => Imaginary.AsSpan(0, _fftSize);

        public void Clear()
        {
            RealSpan.Clear();
            ImaginarySpan.Clear();
        }

        public void Dispose()
        {
            ArrayPool<float>.Shared.Return(Real);
            ArrayPool<float>.Shared.Return(Imaginary);
        }
    }

    public static float[] BuildHannWindow(int length)
    {
        var window = new float[length];
        if (length <= 0)
        {
            return window;
        }

        if (length == 1)
        {
            window[0] = 1f;
            return window;
        }

        // Match torch.hann_window(periodic=true), which is commonly used by STFT frontends.
        var scale = 2f * MathF.PI / length;
        for (var i = 0; i < length; i++)
        {
            window[i] = 0.5f - 0.5f * MathF.Cos(scale * i);
        }

        return window;
    }

    public static float[] ReflectPad(ReadOnlySpan<float> source, int left, int right)
    {
        var padLeft = Math.Max(0, left);
        var padRight = Math.Max(0, right);

        if (padLeft == 0 && padRight == 0)
        {
            return source.ToArray();
        }

        if (source.Length == 0)
        {
            return new float[padLeft + padRight];
        }

        var output = new float[padLeft + source.Length + padRight];
        for (var i = 0; i < output.Length; i++)
        {
            var sourceIndex = ReflectIndex(i - padLeft, source.Length);
            output[i] = source[sourceIndex];
        }

        return output;
    }

    public static ComplexSpectrogram Stft(
        ReadOnlySpan<float> input,
        int nFft,
        int hopLength,
        int winLength,
        ReadOnlySpan<float> window,
        bool center)
    {
        ValidateFftAndWindowArgs(nFft, winLength, window);

        var effectiveHop = Math.Max(1, hopLength);
        var sourceArray = center ? ReflectPad(input, nFft / 2, nFft / 2) : null;
        var source = sourceArray is null ? input : sourceArray.AsSpan();
        var frameCount = source.Length >= nFft
            ? 1 + (source.Length - nFft) / effectiveHop
            : 1;

        var bins = nFft / 2 + 1;
        var real = new float[bins * frameCount];
        var imag = new float[bins * frameCount];

        if (ShouldUseParallel(frameCount))
        {
            var parallelSource = sourceArray ?? input.ToArray();
            var parallelWindow = window[..winLength].ToArray();
            ProcessStftParallel(parallelSource, nFft, winLength, effectiveHop, parallelWindow, bins, frameCount, real, imag);
        }
        else
        {
            ProcessStftSequential(source, nFft, winLength, effectiveHop, window[..winLength], bins, frameCount, real, imag);
        }

        return new ComplexSpectrogram(real, imag, bins, frameCount);
    }

    public static float[] Istft(
        ReadOnlySpan<float> onesidedReal,
        ReadOnlySpan<float> onesidedImag,
        int bins,
        int frames,
        int nFft,
        int hopLength,
        int winLength,
        ReadOnlySpan<float> window,
        bool center,
        int expectedLength)
    {
        ValidateIstftArgs(onesidedReal, onesidedImag, bins, frames, nFft, winLength, window);

        var effectiveHop = Math.Max(1, hopLength);
        var outputLength = nFft + Math.Max(0, frames - 1) * effectiveHop;
        var output = new float[outputLength];
        var windowSumSquare = new float[outputLength];
        var analysisWindow = window[..winLength];

        using var frameBuffer = new PooledFrameBuffer(nFft);
        var frameReal = frameBuffer.RealSpan;
        var frameImag = frameBuffer.ImaginarySpan;

        for (var frame = 0; frame < frames; frame++)
        {
            frameBuffer.Clear();

            for (var bin = 0; bin < bins; bin++)
            {
                var src = bin * frames + frame;
                frameReal[bin] = onesidedReal[src];
                frameImag[bin] = onesidedImag[src];
            }

            for (var bin = bins; bin < nFft; bin++)
            {
                var mirrored = nFft - bin;
                frameReal[bin] = frameReal[mirrored];
                frameImag[bin] = -frameImag[mirrored];
            }

            Radix2Fft.Transform(frameReal, frameImag, inverse: true);

            var frameStart = frame * effectiveHop;
            var copyLength = Math.Max(0, Math.Min(winLength, outputLength - frameStart));
            for (var i = 0; i < copyLength; i++)
            {
                var weight = analysisWindow[i];
                var position = frameStart + i;
                output[position] += frameReal[i] * weight;
                windowSumSquare[position] += weight * weight;
            }
        }

        const float epsilon = 1e-8f;
        for (var i = 0; i < output.Length; i++)
        {
            if (windowSumSquare[i] > epsilon)
            {
                output[i] /= windowSumSquare[i];
            }
        }

        return TrimCentered(output, nFft, center, expectedLength);
    }

    private static void ProcessStftSequential(
        ReadOnlySpan<float> source,
        int nFft,
        int winLength,
        int hopLength,
        ReadOnlySpan<float> window,
        int bins,
        int frameCount,
        float[] outputReal,
        float[] outputImag)
    {
        using var frameBuffer = new PooledFrameBuffer(nFft);

        for (var frame = 0; frame < frameCount; frame++)
        {
            frameBuffer.Clear();
            var frameStart = frame * hopLength;
            var available = Math.Max(0, Math.Min(winLength, source.Length - frameStart));
            if (available > 0)
            {
                MultiplyWindow(source, frameStart, window, frameBuffer.RealSpan, available);
            }

            Radix2Fft.Transform(frameBuffer.RealSpan, frameBuffer.ImaginarySpan, inverse: false);
            StoreSpectrum(frameBuffer.RealSpan, frameBuffer.ImaginarySpan, bins, frameCount, frame, outputReal, outputImag);
        }
    }

    private static void ProcessStftParallel(
        float[] source,
        int nFft,
        int winLength,
        int hopLength,
        float[] window,
        int bins,
        int frameCount,
        float[] outputReal,
        float[] outputImag)
    {
        Parallel.For(
            0,
            frameCount,
            () => new PooledFrameBuffer(nFft),
            (frame, _, local) =>
            {
                local.Clear();
                var frameStart = frame * hopLength;
                var available = Math.Max(0, Math.Min(winLength, source.Length - frameStart));
                if (available > 0)
                {
                    MultiplyWindow(source, frameStart, window, local.RealSpan, available);
                }

                Radix2Fft.Transform(local.RealSpan, local.ImaginarySpan, inverse: false);
                StoreSpectrum(local.RealSpan, local.ImaginarySpan, bins, frameCount, frame, outputReal, outputImag);
                return local;
            },
            local => local.Dispose());
    }

    private static void StoreSpectrum(
        ReadOnlySpan<float> frameReal,
        ReadOnlySpan<float> frameImag,
        int bins,
        int frameCount,
        int frame,
        float[] outputReal,
        float[] outputImag)
    {
        var offset = frame;
        for (var bin = 0; bin < bins; bin++)
        {
            var dst = bin * frameCount + offset;
            outputReal[dst] = frameReal[bin];
            outputImag[dst] = frameImag[bin];
        }
    }

    private static bool ShouldUseParallel(int frameCount)
    {
        return frameCount >= ParallelFrameThreshold && Environment.ProcessorCount > 1;
    }

    private static void ValidateFftAndWindowArgs(int nFft, int winLength, ReadOnlySpan<float> window)
    {
        if (nFft < 2)
        {
            throw new ArgumentException($"FFT size must be >= 2. Got {nFft}.");
        }

        if (!Radix2Fft.IsPowerOfTwo(nFft))
        {
            throw new ArgumentException($"FFT size must be power-of-two. Got {nFft}.");
        }

        if (winLength < 1)
        {
            throw new ArgumentException($"Window length must be >= 1. Got {winLength}.");
        }

        if (winLength > nFft)
        {
            throw new ArgumentException("Window length must be <= FFT size.");
        }

        if (window.Length < winLength)
        {
            throw new ArgumentException("Window buffer is smaller than window length.");
        }
    }

    private static void ValidateIstftArgs(
        ReadOnlySpan<float> onesidedReal,
        ReadOnlySpan<float> onesidedImag,
        int bins,
        int frames,
        int nFft,
        int winLength,
        ReadOnlySpan<float> window)
    {
        ValidateFftAndWindowArgs(nFft, winLength, window);

        if (frames < 1)
        {
            throw new ArgumentException($"Frame count must be >= 1. Got {frames}.");
        }

        var expectedBins = nFft / 2 + 1;
        if (bins != expectedBins)
        {
            throw new ArgumentException($"Bins do not match FFT size. Expected {expectedBins}, got {bins}.");
        }

        var expectedValues = bins * frames;
        if (onesidedReal.Length < expectedValues || onesidedImag.Length < expectedValues)
        {
            throw new ArgumentException("One-sided complex buffers are smaller than expected bins*frames.");
        }
    }

    private static float[] TrimCentered(float[] source, int nFft, bool center, int expectedLength)
    {
        var leftTrim = center ? nFft / 2 : 0;
        var rightTrim = center ? nFft / 2 : 0;
        var available = Math.Max(0, source.Length - leftTrim - rightTrim);
        var resultLength = Math.Max(0, expectedLength > 0 ? expectedLength : available);
        var result = new float[resultLength];

        var copyLength = Math.Min(resultLength, available);
        if (copyLength > 0)
        {
            Array.Copy(source, leftTrim, result, 0, copyLength);
        }

        return result;
    }

    private static int ReflectIndex(int index, int length)
    {
        if (length <= 1)
        {
            return 0;
        }

        while (index < 0 || index >= length)
        {
            if (index < 0)
            {
                index = -index;
            }
            else
            {
                index = 2 * length - index - 2;
            }
        }

        return index;
    }

    private static void MultiplyWindow(
        ReadOnlySpan<float> source,
        int sourceOffset,
        ReadOnlySpan<float> window,
        Span<float> destination,
        int count)
    {
        var simd = Vector<float>.Count;
        var i = 0;
        for (; i <= count - simd; i += simd)
        {
            var s = new Vector<float>(source.Slice(sourceOffset + i));
            var w = new Vector<float>(window.Slice(i));
            (s * w).CopyTo(destination.Slice(i));
        }

        for (; i < count; i++)
        {
            destination[i] = source[sourceOffset + i] * window[i];
        }
    }
}
