using HifiSampler.Core.Audio;
using HifiSampler.Core.HnSep;
using HifiSampler.Core.Stft;
using HifiSampler.Core.Utils;
using System.Numerics;

namespace HifiSampler.Core.Resampler.Pipeline;

public sealed class FeatureExtract(
    ResamplerCacheManager cacheManager,
    PitchAdjustableMelSpectrogram melAnalyzer,
    IHnSep hnsep,
    ResamplerConfig config)
{
    public async Task<(float[,] mel, float scale)> GetFeatureAsync(
        string inputFile,
        ResamplerFlags flags,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (!cacheManager.ShouldBypassCache(flags))
        {
            var cached = await cacheManager.TryLoadMelAsync(inputFile, flags, cancellationToken).ConfigureAwait(false);
            if (cached is not null)
            {
                return cached.Value;
            }
        }

        var generated = await GenerateFeatureAsync(inputFile, flags, cancellationToken).ConfigureAwait(false);
        await cacheManager.SaveMelAsync(inputFile, flags, generated, cancellationToken).ConfigureAwait(false);
        return generated;
    }

    private async Task<(float[,] mel, float scale)> GenerateFeatureAsync(
        string inputFile,
        ResamplerFlags flags,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var audio = AudioIO.ReadMono(inputFile, out _, config.SampleRate);

        if (NeedHnSepSeparation(flags))
        {
            var separated = await GetOrCreateHnSepAsync(inputFile, audio, flags, cancellationToken).ConfigureAwait(false);
            audio = ApplyHnSepFlags(audio, separated, flags);
        }
        else if (flags.Hb != 100 || flags.Hv != 100)
        {
            audio = ApplySimpleScaling(audio, flags.Hb);
        }

        var waveMax = audio.Length == 0 ? 0f : audio.Max(static x => MathF.Abs(x));
        var scale = 1f;
        if (waveMax >= 0.5f)
        {
            scale = 0.5f / waveMax;
            for (var i = 0; i < audio.Length; i++)
            {
                audio[i] *= scale;
            }
        }

        var keyShift = flags.g / 100f;
        var mel = melAnalyzer.Extract(audio, keyShift, 1f);
        ApplyDynamicRangeCompressionInPlace(mel);
        return (mel, scale);
    }

    private async Task<float[]> GetOrCreateHnSepAsync(
        string inputFile,
        float[] audio,
        ResamplerFlags flags,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!cacheManager.ShouldBypassCache(flags))
        {
            var cached = await cacheManager.TryLoadHnSepAsync(inputFile, cancellationToken).ConfigureAwait(false);
            if (cached is not null && cached.Length == audio.Length)
            {
                return cached;
            }
        }

        var separated = hnsep.PredictFromAudio(audio);
        await cacheManager.SaveHnSepAsync(inputFile, separated, cancellationToken).ConfigureAwait(false);
        return separated;
    }

    private float[] ApplyHnSepFlags(float[] original, float[] separated, ResamplerFlags flags)
    {
        var length = Math.Min(original.Length, separated.Length);
        if (length <= 0)
        {
            return [];
        }

        var hb = Math.Clamp(flags.Hb, 0, 500) / 100f;
        var hv = Math.Clamp(flags.Hv, 0, 150) / 100f;
        var result = new float[length];

        if (flags.Ht != 0)
        {
            var tension = Math.Clamp(flags.Ht, -100, 100);
            var tensionScale = -tension / 50f;
            var voiced = new float[length];
            for (var i = 0; i < length; i++)
            {
                voiced[i] = hv * separated[i];
            }

            var filtered = PreEmphasisBaseTension(voiced, tensionScale);
            for (var i = 0; i < length; i++)
            {
                var breath = original[i] - separated[i];
                result[i] = hb * breath + filtered[i];
            }
        }
        else
        {
            for (var i = 0; i < length; i++)
            {
                var voice = separated[i];
                var breath = original[i] - voice;
                result[i] = hb * breath + hv * voice;
            }
        }

        return result;
    }

    private float[] PreEmphasisBaseTension(float[] wave, float b)
    {
        if (wave.Length == 0)
        {
            return [];
        }

        var originalLength = wave.Length;
        var hopSize = Math.Max(1, config.HopSize);
        var padLength = (hopSize - (originalLength % hopSize)) % hopSize;
        var paddedLength = originalLength + padLength;
        var paddedWave = new float[paddedLength];
        Array.Copy(wave, paddedWave, originalLength);

        var nFft = Math.Max(2, config.NFft);
        var winSize = Math.Clamp(config.WinSize, 1, nFft);
        var window = StftEngine.BuildHannWindow(winSize);
        var spectrum = StftEngine.Stft(
            paddedWave,
            nFft,
            hopSize,
            winSize,
            window,
            center: true);

        var bins = spectrum.GetLength(0);
        var frames = spectrum.GetLength(1);
        var sampleRateHalf = Math.Max(1f, config.SampleRate / 2f);
        var x0 = bins / (sampleRateHalf / 1500f);
        if (x0 <= 0f)
        {
            x0 = 1f;
        }

        const float epsilon = 1e-9f;
        for (var bin = 0; bin < bins; bin++)
        {
            var freqFilter = (-b / x0) * bin + b;
            var gainDb = Math.Clamp(freqFilter, -2f, 2f);
            var gainScale = MathF.Exp(gainDb);

            for (var frame = 0; frame < frames; frame++)
            {
                var sample = spectrum[bin, frame];
                var real = (float)sample.Real;
                var imag = (float)sample.Imaginary;
                var amp = MathF.Sqrt(real * real + imag * imag);
                var adjustedAmp = MathF.Max(epsilon, amp) * gainScale;

                if (amp > epsilon)
                {
                    var scale = adjustedAmp / amp;
                    spectrum[bin, frame] = new Complex(real * scale, imag * scale);
                }
                else
                {
                    spectrum[bin, frame] = new Complex(adjustedAmp, 0f);
                }
            }
        }

        var filteredWave = StftEngine.Istft(
            spectrum,
            nFft,
            hopSize,
            winSize,
            window,
            center: true,
            expectedLength: paddedLength);

        var originalMax = 0f;
        for (var i = 0; i < paddedWave.Length; i++)
        {
            originalMax = MathF.Max(originalMax, MathF.Abs(paddedWave[i]));
        }

        var filteredMax = 0f;
        for (var i = 0; i < filteredWave.Length; i++)
        {
            filteredMax = MathF.Max(filteredMax, MathF.Abs(filteredWave[i]));
        }

        var tensionBoost = Math.Clamp(b / -15f, 0f, 0.33f) + 1f;
        var normalization = originalMax > epsilon && filteredMax > epsilon
            ? (originalMax / filteredMax) * tensionBoost
            : 0f;
        for (var i = 0; i < filteredWave.Length; i++)
        {
            filteredWave[i] *= normalization;
        }

        if (filteredWave.Length == originalLength)
        {
            return filteredWave;
        }

        var result = new float[originalLength];
        Array.Copy(filteredWave, result, Math.Min(originalLength, filteredWave.Length));
        return result;
    }

    private float[] ApplySimpleScaling(float[] wave, int breath)
    {
        var scale = Math.Clamp(breath, 0, 500) / 100f;
        var result = new float[wave.Length];
        for (var i = 0; i < wave.Length; i++)
        {
            result[i] = wave[i] * scale;
        }

        return result;
    }

    private void ApplyDynamicRangeCompressionInPlace(float[,] mel)
    {
        const float epsilon = 1e-9f;
        var rows = mel.GetLength(0);
        var cols = mel.GetLength(1);
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                mel[r, c] = MathF.Log(MathF.Max(epsilon, mel[r, c]));
            }
        }
    }

    private bool NeedHnSepSeparation(ResamplerFlags flags) =>
        flags.Ht != 0 || flags.Hb != flags.Hv;
}
