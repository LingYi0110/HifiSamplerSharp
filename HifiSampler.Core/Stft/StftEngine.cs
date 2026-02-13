using System.Numerics;

namespace HifiSampler.Core.Stft;

internal readonly record struct ComplexSpectrogram(
    float[] Real,
    float[] Imaginary,
    int Bins,
    int Frames);

internal static class StftEngine
{
    private sealed class FftFrameBuffer(int fftSize)
    {
        public float[] Real { get; } = new float[fftSize];
        public float[] Imaginary { get; } = new float[fftSize];
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
        if (source.Length == 0)
        {
            return new float[Math.Max(0, left + right)];
        }

        var output = new float[Math.Max(0, left + source.Length + right)];
        for (var i = 0; i < output.Length; i++)
        {
            var sourceIndex = ReflectIndex(i - left, source.Length);
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
        if (nFft < 2)
        {
            throw new ArgumentException($"FFT size must be >= 2. Got {nFft}.");
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

        var source = center ? ReflectPad(input, nFft / 2, nFft / 2) : input.ToArray();
        var windowArray = window[..winLength].ToArray();
        var effectiveHop = Math.Max(1, hopLength);
        var frameCount = source.Length >= nFft
            ? 1 + (source.Length - nFft) / effectiveHop
            : 1;

        var bins = nFft / 2 + 1;
        var real = new float[bins * frameCount];
        var imag = new float[bins * frameCount];

        Parallel.For(
            0,
            frameCount,
            () => new FftFrameBuffer(nFft),
            (frame, _, local) =>
            {
                var frameStart = frame * effectiveHop;
                Array.Clear(local.Real, 0, nFft);
                Array.Clear(local.Imaginary, 0, nFft);

                var available = Math.Max(0, Math.Min(winLength, source.Length - frameStart));
                if (available > 0)
                {
                    MultiplyWindow(source, frameStart, windowArray, local.Real, available);
                }

                FftDispatcher.Transform(local.Real, local.Imaginary, inverse: false);

                var offset = frame;
                for (var bin = 0; bin < bins; bin++)
                {
                    var dst = bin * frameCount + offset;
                    real[dst] = local.Real[bin];
                    imag[dst] = local.Imaginary[bin];
                }

                return local;
            },
            _ => { });

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
        if (nFft < 2)
        {
            throw new ArgumentException($"FFT size must be >= 2. Got {nFft}.");
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

        var effectiveHop = Math.Max(1, hopLength);
        var outputLength = nFft + Math.Max(0, frames - 1) * effectiveHop;
        var output = new float[outputLength];
        var windowSumSquare = new float[outputLength];

        var frameReal = new float[nFft];
        var frameImag = new float[nFft];

        for (var frame = 0; frame < frames; frame++)
        {
            Array.Clear(frameReal, 0, nFft);
            Array.Clear(frameImag, 0, nFft);

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

            FftDispatcher.Transform(frameReal, frameImag, inverse: true);

            var frameStart = frame * effectiveHop;
            var copyLength = Math.Max(0, Math.Min(winLength, outputLength - frameStart));
            for (var i = 0; i < copyLength; i++)
            {
                var weight = window[i];
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
        float[] source,
        int sourceOffset,
        float[] window,
        float[] destination,
        int count)
    {
        var simd = Vector<float>.Count;
        var i = 0;
        for (; i <= count - simd; i += simd)
        {
            var s = new Vector<float>(source, sourceOffset + i);
            var w = new Vector<float>(window, i);
            (s * w).CopyTo(destination, i);
        }

        for (; i < count; i++)
        {
            destination[i] = source[sourceOffset + i] * window[i];
        }
    }
}
