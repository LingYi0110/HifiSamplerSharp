using System.Text.Json.Serialization;

namespace HifiSampler.Client;

[JsonSourceGenerationOptions(PropertyNameCaseInsensitive = false)]
[JsonSerializable(typeof(ResamplerRequest))]
internal sealed partial class ClientJsonContext : JsonSerializerContext;

