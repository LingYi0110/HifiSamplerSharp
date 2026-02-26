using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace HifiSampler.Core.Vocoder;

public sealed class OnnxVocoder : IVocoder, IDisposable
{
    private readonly InferenceSession _session;
    private readonly int _numMels;
    private readonly int _hopSize;

    public OnnxVocoder(string modelPath, string device, int deviceId = 0, int numMels = 128, int hopSize = 512)
    {
        _numMels = numMels;
        _hopSize = hopSize;
        _session = Utils.OnnxUtils.CreateSession(modelPath, device, deviceId);
    }
    
    public float[] SpecToWav(float[,] mel, float[] f0)
    {
        // Prepare mel tensor
        // [1, time, num_mels]
        int nMels = mel.GetLength(0);
        int timeFrames = mel.GetLength(1);
        var melTensor = new DenseTensor<float>(new[] { 1, timeFrames, nMels });
        var melSpan = melTensor.Buffer.Span;
        int idx = 0;
        for (int t = 0; t < timeFrames; t++)
        {
            for (int m = 0; m < nMels; m++)
            {
                melSpan[idx++] = mel[m, t];
            }
        }
        
        // Prepare f0 tensor
        // [1, time]
        var f0Tensor = new DenseTensor<float>(f0, new[] { 1, f0.Length });
        
        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("mel", melTensor),
            NamedOnnxValue.CreateFromTensor("f0", f0Tensor)
        };
        
        using var results = _session.Run(inputs);
        var output = results[0].AsTensor<float>();
        
        return output.ToArray();
    }
    
    public void Dispose()
    {
        _session.Dispose();
    }
}
