namespace HifiSampler.Core.Resampler;

public sealed class ResamplerConfig
{
    public const string SectionName = "Sampler";

    public int Port { get; set; } = 8572;
    public string? CachePath { get; set; }
    public int MaxWorkers { get; set; } = 2;
    public int SampleRate { get; set; } = 44100;
    public int HopSize { get; set; } = 512;
    public int OriginHopSize { get; set; } = 128;
    public int NFft { get; set; } = 2048;
    public int WinSize { get; set; } = 2048;
    public int NumMels { get; set; } = 128;
    public float MelFMin { get; set; } = 40f;
    public float MelFMax { get; set; } = 16000f;

    public int Fill { get; set; } = 8;
    public double PeakLimit { get; set; } = 0.9;
    public bool WaveNorm { get; set; } = true;
    public bool LoopMode { get; set; }

    public HnSepConfig HnSepConfig { get; set; } = new();
    public VocoderConfig VocoderConfig { get; set; } = new();
}

public sealed class VocoderConfig
{
    public string ModelType { get; set; } = "onnx";
    public string ModelPath { get; set; } = string.Empty;
    public string Device { get; set; } = "cpu";
    public int DeviceId { get; set; } = 0;
    public int NumMels { get; set; } = 128;
}

public sealed class HnSepConfig
{
    public string ModelType { get; set; } = "onnx";
    public string ModelPath { get; set; } = string.Empty;
    public string Device { get; set; } = "cpu";
    public int DeviceId { get; set; } = 0;
    public int HopLength { get; set; } = 512;
    public int NFft { get; set; } = 2048;
}
