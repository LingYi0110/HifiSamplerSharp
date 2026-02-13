using HifiSampler.Core.Resampler;

namespace HifiSampler.Core.Vocoder;

public static class VocoderFactory
{
    public static IVocoder Create(VocoderConfig config)
    {
        switch (config.ModelType)
        {
            case "onnx":
                return new OnnxVocoder(config.ModelPath, config.Device, config.DeviceId, config.NumMels);
            default:
                throw new ArgumentOutOfRangeException(nameof(config.ModelType), config.ModelType, null);
        }
    }
}
