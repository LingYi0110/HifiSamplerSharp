using HifiSampler.Core.Utils;

namespace HifiSampler.Core.Vocoder;

public interface IVocoder
{
    float[] SpecToWav(FloatMatrix mel, float[] f0);
}
