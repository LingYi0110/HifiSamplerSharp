namespace HifiSampler.Core.Vocoder;

public interface IVocoder
{
    float[] SpecToWav(float[,] mel, float[] f0);
}
