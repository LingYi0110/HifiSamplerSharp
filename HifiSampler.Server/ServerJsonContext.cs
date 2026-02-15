using System.Text.Json.Serialization;
using HifiSampler.Server.Contracts;

namespace HifiSampler.Server;

[JsonSourceGenerationOptions(PropertyNameCaseInsensitive = false)]
[JsonSerializable(typeof(ResamplerRequestDto))]
internal sealed partial class ServerJsonContext : JsonSerializerContext;
