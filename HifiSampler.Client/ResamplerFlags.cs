namespace HifiSampler.Client;

public sealed record ResamplerFlags
{
    public int g { get; init; }
    public int Hb { get; init; } = 100;
    public int Hv { get; init; } = 100;
    public int HG { get; init; }
    public int P { get; init; } = 100;
    public int t { get; init; }
    public int Ht { get; init; }
    public int A { get; init; }
    public bool G { get; init; }
    public bool He { get; init; }
}