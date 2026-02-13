using HifiSampler.Core.HnSep;
using HifiSampler.Core.Pipeline;
using HifiSampler.Core.Utils;
using HifiSampler.Core.Vocoder;

namespace HifiSampler.Core.Resampler;

public static class ResamplerFactory
{
    public static Resampler CreateResampler(ResamplerConfig config)
    {
        var cache = new ResamplerCacheManager();
        var melAnalyser = new PitchAdjustableMelSpectrogram(
            sampleRate: config.SampleRate,
            nFft: config.NFft,
            winLength: config.WinSize,
            hopLength: config.OriginHopSize,
            fMin: config.MelFMin,
            fMax: config.MelFMax,
            nMels: config.NumMels);
        var hnsep = HnSepFactory.Create(config.HnSepConfig);
        var featurePipeline = new FeatureExtract(cache, melAnalyser, hnsep, config);

        var vocoder = VocoderFactory.Create(config.VocoderConfig);
        return new Resampler(featurePipeline, vocoder, config);
    }
}
