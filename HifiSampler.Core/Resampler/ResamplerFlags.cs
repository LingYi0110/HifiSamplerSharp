namespace HifiSampler.Core.Resampler;

/// <summary>
/// Configuration flags for the resampler with various audio processing parameters.
/// </summary>
public sealed record ResamplerFlags
{
    /// <summary>
    /// Adjust gender/formants.
    /// Range: -600 to 600 | Default: 0
    /// </summary>
    public int g { get; init; }

    /// <summary>
    /// Adjust breath/noise.
    /// Range: 0 to 500 | Default: 100
    /// </summary>
    public int Hb { get; init; } = 100;

    /// <summary>
    /// Adjust voice/harmonic.
    /// Range: 0 to 150 | Default: 100
    /// </summary>
    public int Hv { get; init; } = 100;

    /// <summary>
    /// Vocal fry/growl.
    /// Range: 0 to 100 | Default: 0
    /// </summary>
    public int HG { get; init; }

    /// <summary>
    /// Normalize loudness at the note level, targeting -16 LUFS. 
    /// Enable this by setting wave_norm to true in your config.yaml file.
    /// Range: 0 to 100 | Default: 100
    /// </summary>
    public int P { get; init; } = 100;

    /// <summary>
    /// Shift the pitch by a specific amount, in cents. 1 cent = 1/100 of a semitone.
    /// Range: -1200 to 1200 | Default: 0
    /// </summary>
    public int t { get; init; }

    /// <summary>
    /// Adjust tension.
    /// Range: -100 to 100 | Default: 0
    /// </summary>
    public int Ht { get; init; }

    /// <summary>
    /// Modulating the amplitude based on pitch variations, which helps creating a more realistic vibrato.
    /// Range: -100 to 100 | Default: 0
    /// </summary>
    public int A { get; init; }

    /// <summary>
    /// Force to regenerate feature cache (Ignoring existed cache). No value needed.
    /// </summary>
    public bool G { get; init; }
    
    /// <summary>
    /// Enable Mel spectrum loop mode. No value needed.
    /// </summary>
    public bool He { get; init; }
}