using HifiSampler.Core.Resampler;

namespace HifiSampler.Core.HnSep;

public class HnSepFactory
{
    public static IHnSep Create(HnSepConfig config)
    {
        switch (config.ModelType)
        {
            case "onnx":
                return new HnSepModel(config.ModelPath, config.Device, config.DeviceId, config.NFft, config.HopLength);
            default:
                throw new ArgumentOutOfRangeException(nameof(config.ModelType), config.ModelType, null);
        }
    }
}