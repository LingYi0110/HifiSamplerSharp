using HifiSampler.Core.Audio;
using HifiSampler.Core.HnSep;
using HifiSampler.Core.Resampler;
using HifiSampler.Core.Utils;

namespace HifiSampler.Core.Pipeline;

public sealed class FeatureExtract(
    ResamplerCacheManager cacheManager,
    PitchAdjustableMelSpectrogram melAnalyzer,
    IHnSep hnsep,
    ResamplerConfig config)
{
    public async Task<(FloatMatrix mel, float scale)> GetFeatureAsync(
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

    private async Task<(FloatMatrix mel, float scale)> GenerateFeatureAsync(
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

    private static float[] ApplyHnSepFlags(float[] original, float[] separated, ResamplerFlags flags)
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

    private static float[] PreEmphasisBaseTension(float[] input, float tension)
    {
        // Lightweight approximation: match Python behavior directionally without heavy STFT dependency.
        var result = new float[input.Length];
        var lowBlend = Math.Clamp(tension / 2f, -1f, 1f);
        float prev = 0f;
        for (var i = 0; i < input.Length; i++)
        {
            var high = input[i] - 0.95f * prev;
            prev = input[i];
            result[i] = input[i] + lowBlend * high;
        }

        return result;
    }

    private static float[] ApplySimpleScaling(float[] wave, int breath)
    {
        var scale = Math.Clamp(breath, 0, 500) / 100f;
        var result = new float[wave.Length];
        for (var i = 0; i < wave.Length; i++)
        {
            result[i] = wave[i] * scale;
        }

        return result;
    }

    private static void ApplyDynamicRangeCompressionInPlace(FloatMatrix mel)
    {
        const float epsilon = 1e-9f;
        var rows = mel.Rows;
        var cols = mel.Cols;
        for (var r = 0; r < rows; r++)
        {
            var row = mel.RowSpan(r);
            for (var c = 0; c < cols; c++)
            {
                row[c] = MathF.Log(MathF.Max(epsilon, row[c]));
            }
        }
    }

    private static bool NeedHnSepSeparation(ResamplerFlags flags) =>
        flags.Ht != 0 || flags.Hb != flags.Hv;
}
