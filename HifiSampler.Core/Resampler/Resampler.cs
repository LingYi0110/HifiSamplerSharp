using HifiSampler.Core.Audio;
using HifiSampler.Core.Pipeline;
using HifiSampler.Core.Vocoder;

namespace HifiSampler.Core.Resampler;

public sealed class Resampler(
    FeatureExtract featureExtract,
    IVocoder vocoder,
    ResamplerConfig config)
{
    public async Task<ResampleResult> RenderAsync(
        string inputFile,
        string outputFile,
        int pitchMidi,
        double velocity,
        ResamplerFlags resamplerFlags,
        double offset,
        int length,
        double consonant,
        double cutoff,
        double volume,
        double modulation,
        double tempo,
        double[] pitchBendCents,
        CancellationToken cancellationToken)
    {
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            var flags = ClipFlags(resamplerFlags);
            var (melOrigin, scale) = await featureExtract.GetFeatureAsync(inputFile, flags, cancellationToken)
                .ConfigureAwait(false);

            if (string.Equals(outputFile, "nul", StringComparison.OrdinalIgnoreCase))
            {
                return new ResampleResult(200, "Success: null output skipped.");
            }

            var thopOrigin = (double)config.OriginHopSize / config.SampleRate;
            var thop = (double)config.HopSize / config.SampleRate;

            var tAreaOrigin = BuildTimeAxis(melOrigin.GetLength(1), thopOrigin);
            var totalTime = tAreaOrigin[^1] + thopOrigin / 2;

            var vel = Math.Pow(2, 1 - velocity / 100.0);
            var start = offset / 1000.0;
            var cutoffSeconds = cutoff / 1000.0;
            var end = cutoff < 0 ? start - cutoffSeconds : totalTime - cutoffSeconds;
            var con = start + consonant / 1000.0;
            var lengthReq = length / 1000.0;
            var stretchLength = end - con;

            if (config.LoopMode || flags.He)
            {
                (melOrigin, tAreaOrigin, stretchLength) =
                    BuildLoopedMel(melOrigin, tAreaOrigin, con, end, thopOrigin, lengthReq);
                totalTime = tAreaOrigin[^1] + thopOrigin / 2;
            }

            var scalingRatio = stretchLength < lengthReq && stretchLength > 1e-8
                ? lengthReq / stretchLength
                : 1.0;

            var stretchedNFrames = (int)((con * vel + (totalTime - con) * scalingRatio) / thop) + 1;
            var stretchedTMel = Enumerable.Range(0, (int)stretchedNFrames)
                .Select(i => i * thop + thop / 2)
                .ToArray();

            var startLeftMelFrames = (int)((start * vel + thop / 2) / thop);
            var cutLeftMelFrames = Math.Max(0, startLeftMelFrames - config.Fill);
            var endRightMelFrames = stretchedNFrames - (int)((lengthReq + con * vel + thop / 2) / thop);
            var cutRightMelFrames = Math.Max(0, endRightMelFrames - config.Fill);

            var left = Math.Min(cutLeftMelFrames, stretchedTMel.Length);
            var right = Math.Min(cutRightMelFrames, Math.Max(0, stretchedTMel.Length - left));
            var trimmed = stretchedTMel.Skip(left).Take(stretchedTMel.Length - left - right).ToArray();

            var stretchTMel = trimmed
                .Select(t => Math.Clamp(TimeStretch.Stretch(t, con, scalingRatio, vel), 0, tAreaOrigin[^1]))
                .ToArray();

            var newStart = start * vel - cutLeftMelFrames * thop;
            var newEnd = (lengthReq + con * vel) - cutLeftMelFrames * thop;

            var melRender = InterpolateMelOverTime(melOrigin, tAreaOrigin, stretchTMel);
            var t = Enumerable.Range(0, melRender.GetLength(1)).Select(i => i * thop).ToArray();

            var f0Render = PitchCurve.Build(pitchMidi, pitchBendCents, flags, tempo, newStart, t)
                .Select(static x => (float)x)
                .ToArray();

            var wavCon = vocoder.SpecToWav(melRender, f0Render);
            var cutL = Math.Max(0, (int)(newStart * config.SampleRate));
            var cutR = Math.Min(wavCon.Length, (int)(newEnd * config.SampleRate));
            if (cutR <= cutL)
            {
                return new ResampleResult(500, "Error processing: invalid render range.");
            }

            var render = wavCon[cutL..cutR];

            if (flags.A != 0)
            {
                render = Modulation.ApplyAmplitudeFromPitch(render, t, f0Render, newStart, newEnd, flags.A);
            }

            if (Math.Abs(scale) > 1e-8f)
            {
                for (var i = 0; i < render.Length; i++)
                {
                    render[i] /= scale;
                }
            }

            if (flags.HG > 0)
            {
                render = Modulation.ApplyGrowl(render, config.SampleRate, flags.HG);
            }

            if (config.WaveNorm && flags.P > 0)
            {
                render = LoudnessNorm.Normalize(render, config.SampleRate, flags.P);
            }

            var peak = render.Length == 0 ? 0f : render.Max(static x => MathF.Abs(x));
            if (peak > config.PeakLimit && peak > 1e-8f)
            {
                var norm = (float)(config.PeakLimit / peak);
                for (var i = 0; i < render.Length; i++)
                {
                    render[i] *= norm;
                }
            }

            var volumeScale = (float)(volume / 100.0);
            for (var i = 0; i < render.Length; i++)
            {
                render[i] *= volumeScale;
            }

            _ = modulation; // parsed for parity, currently no-op by design.
            AudioIO.WriteWavMono(outputFile, render, config.SampleRate);
            return new ResampleResult(200, $"Success: '{Path.GetFileNameWithoutExtension(inputFile)}' -> '{Path.GetFileName(outputFile)}'");
        }
        catch (FileNotFoundException ex)
        {
            return new ResampleResult(404, "Error processing: Input file not found.", ex.ToString());
        }
        catch (Exception ex)
        {
            return new ResampleResult(500, "Error processing: Internal error.", ex.ToString());
        }
    }

    private static double[] BuildTimeAxis(int frameCount, double hopSeconds)
    {
        var t = new double[frameCount];
        for (var i = 0; i < frameCount; i++)
        {
            t[i] = i * hopSeconds + hopSeconds / 2;
        }

        return t;
    }

    private static (float[,] mel, double[] tAreaOrigin, double stretchLength) BuildLoopedMel(
        float[,] melOrigin,
        double[] tAreaOrigin,
        double con,
        double end,
        double thopOrigin,
        double lengthReq)
    {
        var left = (int)((con + thopOrigin / 2) / thopOrigin);
        var right = Math.Max(left + 1, (int)((end + thopOrigin / 2) / thopOrigin));
        left = Math.Clamp(left, 0, melOrigin.GetLength(1) - 1);
        right = Math.Clamp(right, left + 1, melOrigin.GetLength(1));

        var loop = SliceMel(melOrigin, left, right);
        var padLoopSize = (int)(lengthReq / thopOrigin) + 1;
        var padded = ReflectPadMel(loop, padLoopSize);

        var prefix = SliceMel(melOrigin, 0, left);
        var merged = ConcatMel(prefix, padded);
        var newT = BuildTimeAxis(merged.GetLength(1), thopOrigin);
        return (merged, newT, padLoopSize * thopOrigin);
    }

    private static float[,] SliceMel(float[,] mel, int start, int end)
    {
        var rows = mel.GetLength(0);
        var cols = Math.Max(0, end - start);
        var result = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                result[r, c] = mel[r, start + c];
            }
        }

        return result;
    }

    private static float[,] ReflectPadMel(float[,] mel, int padRight)
    {
        var rows = mel.GetLength(0);
        var cols = mel.GetLength(1);
        var outCols = cols + Math.Max(0, padRight);
        var result = new float[rows, outCols];

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                result[r, c] = mel[r, c];
            }

            for (var c = cols; c < outCols; c++)
            {
                var idx = c - cols;
                var period = idx / Math.Max(1, cols);
                var pos = idx % Math.Max(1, cols);
                if (period % 2 == 1)
                {
                    pos = cols - 1 - pos;
                }

                result[r, c] = mel[r, pos];
            }
        }

        return result;
    }

    private static float[,] ConcatMel(float[,] left, float[,] right)
    {
        var rows = left.GetLength(0);
        var colsL = left.GetLength(1);
        var colsR = right.GetLength(1);
        var result = new float[rows, colsL + colsR];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < colsL; c++)
            {
                result[r, c] = left[r, c];
            }

            for (var c = 0; c < colsR; c++)
            {
                result[r, colsL + c] = right[r, c];
            }
        }

        return result;
    }

    private static float[,] InterpolateMelOverTime(float[,] mel, double[] x, double[] targets)
    {
        var rows = mel.GetLength(0);
        var cols = targets.Length;
        var result = new float[rows, cols];

        for (var r = 0; r < rows; r++)
        {
            var y = new double[x.Length];
            for (var c = 0; c < x.Length; c++)
            {
                y[c] = mel[r, c];
            }

            var interp = LinearInterpolate(x, y, targets);
            for (var c = 0; c < cols; c++)
            {
                result[r, c] = (float)interp[c];
            }
        }

        return result;
    }
    private static double[] LinearInterpolate(double[] x, double[] y, double[] target)
    {
        if (x.Length == 0 || y.Length == 0 || x.Length != y.Length)
        {
            return [];
        }

        var result = new double[target.Length];
        var cursor = 0;
        for (var i = 0; i < target.Length; i++)
        {
            var t = target[i];
            while (cursor + 1 < x.Length && x[cursor + 1] < t)
            {
                cursor++;
            }

            if (cursor + 1 >= x.Length)
            {
                result[i] = y[^1];
                continue;
            }

            var t0 = x[cursor];
            var t1 = x[cursor + 1];
            var y0 = y[cursor];
            var y1 = y[cursor + 1];
            var ratio = (t - t0) / Math.Max(t1 - t0, 1e-8);
            result[i] = y0 + (y1 - y0) * ratio;
        }

        return result;
    }

    private static ResamplerFlags ClipFlags(ResamplerFlags flags) =>
        flags with
        {
            g = Math.Clamp(flags.g, -600, 600),
            Hb = Math.Clamp(flags.Hb, 0, 500),
            Hv = Math.Clamp(flags.Hv, 0, 150),
            HG = Math.Clamp(flags.HG, 0, 100),
            P = Math.Clamp(flags.P, 0, 100),
            t = Math.Clamp(flags.t, -1200, 1200),
            Ht = Math.Clamp(flags.Ht, -100, 100),
            A = Math.Clamp(flags.A, -100, 100)
        };
}
