using HifiSampler.Core.Resampler;
using HifiSampler.Core.Utils;

namespace HifiSampler.Core.Pipeline;

public static class PitchCurve
{
    private static double[] MidiToHz(double[] midiValues)
    {
        var result = new double[midiValues.Length];
        for (var i = 0; i < midiValues.Length; i++)
        {
            // f = 440 * 2^((midi - 69) / 12)
            result[i] = 440.0 * Math.Pow(2.0, (midiValues[i] - 69.0) / 12.0);
        }
        return result;
    }

    public static double[] Build(
        int pitchMidi, 
        double[] pitchBendCents, 
        ResamplerFlags flags, 
        double tempo,
        double startTime,
        double[] timeline)
    {
        // 计算基础 pitch 数组 (MIDI 值)
        var pitch = new double[pitchBendCents.Length];
        for (var i = 0; i < pitchBendCents.Length; i++)
        {
            pitch[i] = pitchMidi + (pitchBendCents[i] / 100.0);
            if (flags.t != 0)
            {
                pitch[i] += flags.t / 100.0;
            }
        }
        
        // 计算pitch的时间轴
        var tPitch = new double[pitch.Length];
        for (var i = 0; i < pitch.Length; i++)
        {
            tPitch[i] = 60.0 * i / (tempo * 96.0) + startTime;
        }
        
        var interpolator = new AkimaInterpolator(tPitch, pitch);
        
        var tClipped = new double[timeline.Length];
        var tPitchLast = tPitch[^1];
        for (var i = 0; i < timeline.Length; i++)
        {
            tClipped[i] = Math.Clamp(timeline[i], startTime, tPitchLast);
        }
        
        var f0Render = MidiToHz(interpolator.Interpolate(tClipped));
        
        return f0Render;
    }
}
